use std::time::Duration;

use salvo::prelude::*;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::engines::ascend;
use crate::engines::qwen3_tts;
use crate::engines::tts_trait::{TtsCloneRequest, TtsRequest as TtsTraitRequest, TtsResponse};
use crate::error::render_error;
use crate::inference::{AudioChunk, InferenceRequest, TtsRequest};
use crate::types::{SpeechCloneRequest, SpeechRequest, TranscriptionRequest, VoiceNotFoundError};
use crate::voice_registry::VoiceRegistry;

use super::helpers::{get_state, send_and_wait};

/// Normalize a TTS voice name: treat empty/"default" as the fallback "vivian".
fn normalize_voice(voice: Option<String>) -> String {
    match voice.as_deref() {
        None | Some("") | Some("default") => "vivian".to_string(),
        Some(v) => v.to_string(),
    }
}

/// Timeout for audio transcription
const TRANSCRIPTION_TIMEOUT: Duration = Duration::from_secs(1800); // 30 minutes
/// Timeout for text-to-speech (non-streaming WAV fallback)
const TTS_TIMEOUT: Duration = Duration::from_secs(300); // 5 minutes (pool handles long text)

/// POST /v1/audio/transcriptions - Speech-to-text
#[handler]
pub async fn audio_transcriptions(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    // Use larger body size limit for audio uploads (10MB)
    let request: TranscriptionRequest = req
        .parse_json_with_max_size(10 * 1024 * 1024)
        .await
        .map_err(|e| {
            tracing::error!("Failed to parse request: {}", e);
            StatusError::bad_request()
        })?;

    let response = send_and_wait(
        &state.inference_tx,
        |tx| InferenceRequest::Transcribe { request, expected_backend: None, response_tx: tx },
        TRANSCRIPTION_TIMEOUT,
    )
    .await?;

    res.render(Json(response));
    Ok(())
}

/// POST /v1/audio/speech - Text-to-speech (streaming via TTS pool)
///
/// Returns chunked PCM audio as it's generated. The response is raw PCM
/// (16-bit signed LE, mono, 24kHz) with Transfer-Encoding: chunked.
/// Clients that want WAV can use `?format=wav` to get a complete WAV file.
#[handler]
pub async fn audio_speech(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    let request: SpeechRequest = req.parse_json_with_max_size(10 * 1024 * 1024).await.map_err(|e| {
        tracing::error!("Failed to parse request: {}", e);
        StatusError::bad_request()
    })?;

    let wants_wav = req.query::<String>("format").as_deref() == Some("wav")
        || request.response_format == "wav"
        || request.reference_audio.is_some();

    let chunk_rx = spawn_per_sentence_tts(
        state.inference_tx.clone(),
        qwen3_tts::split_sentences(&request.input),
        normalize_voice(request.voice),
        request.language.unwrap_or_else(|| "chinese".to_string()),
        request.speed,
        request.instruct,
    );

    if wants_wav {
        let pcm_data = collect_pcm_chunks(chunk_rx).await?;
        let wav_data = pcm_to_wav(&pcm_data, 24000);
        res.headers_mut()
            .insert("Content-Type", "audio/wav".parse().unwrap());
        res.write_body(wav_data).ok();
        return Ok(());
    }

    stream_pcm_response(chunk_rx, res);
    Ok(())
}

/// POST /v1/audio/speech/clone - Voice cloning TTS
///
/// Accepts multipart form: `reference_audio` (raw WAV/MP3/OGG file),
/// `input` (text), `language` (optional), `speed` (optional),
/// `instruct`/`prompt` (optional style instruction).
/// Streams PCM chunks per-sentence by default (pseudo-streaming).
/// Use `?format=wav` for a complete WAV response.
#[handler]
pub async fn audio_speech_clone(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    let wants_wav = req.query::<String>("format").as_deref() == Some("wav");
    let request = parse_clone_multipart(req).await?;

    let chunk_rx = spawn_per_sentence_clone(
        state.inference_tx.clone(),
        qwen3_tts::split_sentences(&request.input),
        request.reference_audio,
        request.language,
        request.speed,
        request.instruct,
    );

    if wants_wav {
        let pcm_data = collect_pcm_chunks(chunk_rx).await?;
        let wav_data = pcm_to_wav(&pcm_data, 24000);
        res.headers_mut()
            .insert("Content-Type", "audio/wav".parse().unwrap());
        res.write_body(wav_data).ok();
        return Ok(());
    }

    stream_pcm_response(chunk_rx, res);
    Ok(())
}

// ============================================================================
// Model-specific TTS endpoints
// ============================================================================

/// POST /v1/audio/tts/qwen3 — Qwen3-TTS with preset voices
///
/// Always routes to the TTS pool (CustomVoice model).
/// Streams PCM by default; use `?format=wav` for a complete WAV response.
#[handler]
pub async fn tts_qwen3(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    if state.server_config.is_category_disabled("tts") {
        render_error(
            res,
            salvo::http::StatusCode::FORBIDDEN,
            "TTS is disabled by server configuration",
            "forbidden",
        );
        return Ok(());
    }

    let request: SpeechRequest = req.parse_json_with_max_size(10 * 1024 * 1024).await.map_err(|e| {
        tracing::error!("Failed to parse request: {}", e);
        StatusError::bad_request()
    })?;

    let wants_wav = req.query::<String>("format").as_deref() == Some("wav")
        || request.response_format == "wav";

    // Resolve the requested voice through the same normalization as before
    // (empty/"default" -> "vivian"), then reject unknown names with 404.
    // Silent fallback to a default voice is what prompted this endpoint —
    // fm_tts callers were getting 200 + audio in the wrong voice because
    // an unregistered `voice` string was being swallowed downstream.
    let resolved_voice = normalize_voice(request.voice.clone());
    let registry = VoiceRegistry::load();
    if !registry.contains(&resolved_voice) {
        let requester = request_peer(req);
        let available = registry.available_voices().to_vec();
        tracing::info!(
            "rejected TTS request: unknown voice {} (available: {}) from {}",
            resolved_voice,
            available.join(", "),
            requester,
        );
        let message = format!(
            "Voice '{}' is not registered. Available: {}",
            resolved_voice,
            available.join(", "),
        );
        res.status_code(salvo::http::StatusCode::NOT_FOUND);
        res.render(Json(VoiceNotFoundError {
            error: "voice_not_found",
            message,
            requested_voice: resolved_voice,
            available_voices: available,
        }));
        return Ok(());
    }

    let chunk_rx = spawn_per_sentence_tts(
        state.inference_tx.clone(),
        qwen3_tts::split_sentences(&request.input),
        resolved_voice,
        request.language.unwrap_or_else(|| "chinese".to_string()),
        request.speed,
        request.instruct,
    );

    if wants_wav {
        let pcm_data = collect_pcm_chunks(chunk_rx).await?;
        let wav_data = pcm_to_wav(&pcm_data, 24000);
        res.headers_mut()
            .insert("Content-Type", "audio/wav".parse().unwrap());
        res.write_body(wav_data).ok();
    } else {
        stream_pcm_response(chunk_rx, res);
    }
    Ok(())
}

/// Best-effort client identifier for log lines. Prefers
/// `X-Forwarded-For` / `X-Real-IP` (gateway-provided), else the peer
/// socket. Falls back to "unknown" so logs stay single-line.
fn request_peer(req: &Request) -> String {
    if let Some(fwd) = req.headers().get("x-forwarded-for") {
        if let Ok(s) = fwd.to_str() {
            if let Some(first) = s.split(',').next() {
                let trimmed = first.trim();
                if !trimmed.is_empty() {
                    return trimmed.to_string();
                }
            }
        }
    }
    if let Some(real) = req.headers().get("x-real-ip") {
        if let Ok(s) = real.to_str() {
            if !s.is_empty() {
                return s.to_string();
            }
        }
    }
    req.remote_addr().to_string()
}

/// POST /v1/audio/tts/clone — Qwen3-TTS voice cloning (Base model)
///
/// Dedicated endpoint for zero-shot voice cloning via x-vector speaker embedding.
/// Streams PCM by default; use `?format=wav` for a complete WAV response.
#[handler]
pub async fn tts_clone(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    if state.server_config.is_category_disabled("tts") {
        render_error(
            res,
            salvo::http::StatusCode::FORBIDDEN,
            "TTS is disabled by server configuration",
            "forbidden",
        );
        return Ok(());
    }

    let wants_wav = req.query::<String>("format").as_deref() == Some("wav");
    let request = parse_clone_multipart(req).await?;

    let chunk_rx = spawn_per_sentence_clone(
        state.inference_tx.clone(),
        qwen3_tts::split_sentences(&request.input),
        request.reference_audio,
        request.language,
        request.speed,
        request.instruct,
    );

    if wants_wav {
        let pcm_data = collect_pcm_chunks(chunk_rx).await?;
        let wav_data = pcm_to_wav(&pcm_data, 24000);
        res.headers_mut()
            .insert("Content-Type", "audio/wav".parse().unwrap());
        res.write_body(wav_data).ok();
    } else {
        stream_pcm_response(chunk_rx, res);
    }
    Ok(())
}

/// POST /v1/audio/tts/sovits — GPT-SoVITS text-to-speech
///
/// Routes to the inference thread's GPT-SoVITS engine (not the Qwen3 TTS pool).
/// Returns WAV audio. Requires GPT-SoVITS model to be loaded via
/// `POST /v1/models/load { "model": "<ref_audio>", "model_type": "tts" }`.
#[handler]
pub async fn tts_sovits(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    if state.server_config.is_category_disabled("tts") {
        render_error(
            res,
            salvo::http::StatusCode::FORBIDDEN,
            "TTS is disabled by server configuration",
            "forbidden",
        );
        return Ok(());
    }

    let request: SpeechRequest = req.parse_json_with_max_size(10 * 1024 * 1024).await.map_err(|e| {
        tracing::error!("Failed to parse request: {}", e);
        StatusError::bad_request()
    })?;

    // Route to inference thread (GPT-SoVITS, not TTS pool)
    let audio_data = send_and_wait(
        &state.inference_tx,
        |tx| InferenceRequest::Speech { request, response_tx: tx },
        TTS_TIMEOUT,
    )
    .await?;

    res.headers_mut()
        .insert("Content-Type", "audio/wav".parse().unwrap());
    res.write_body(audio_data).ok();
    Ok(())
}

// ============================================================================
// Model-specific ASR endpoints
// ============================================================================

/// POST /v1/audio/asr/qwen3 — Qwen3-ASR speech recognition
///
/// Routes to the inference thread; validates that Qwen3-ASR is the loaded backend.
#[handler]
pub async fn asr_qwen3(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    if state.server_config.is_category_disabled("asr") {
        render_error(
            res,
            salvo::http::StatusCode::FORBIDDEN,
            "ASR is disabled by server configuration",
            "forbidden",
        );
        return Ok(());
    }

    let request: TranscriptionRequest = req
        .parse_json_with_max_size(10 * 1024 * 1024)
        .await
        .map_err(|e| {
            tracing::error!("Failed to parse request: {}", e);
            StatusError::bad_request()
        })?;

    let response = send_and_wait(
        &state.inference_tx,
        |tx| InferenceRequest::Transcribe {
            request,
            expected_backend: Some("qwen3-asr".to_string()),
            response_tx: tx,
        },
        TRANSCRIPTION_TIMEOUT,
    )
    .await?;

    res.render(Json(response));
    Ok(())
}

/// POST /v1/audio/asr/paraformer — Paraformer speech recognition
///
/// Routes to the inference thread; validates that Paraformer is the loaded backend.
#[handler]
pub async fn asr_paraformer(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    if state.server_config.is_category_disabled("asr") {
        render_error(
            res,
            salvo::http::StatusCode::FORBIDDEN,
            "ASR is disabled by server configuration",
            "forbidden",
        );
        return Ok(());
    }

    let request: TranscriptionRequest = req
        .parse_json_with_max_size(10 * 1024 * 1024)
        .await
        .map_err(|e| {
            tracing::error!("Failed to parse request: {}", e);
            StatusError::bad_request()
        })?;

    let response = send_and_wait(
        &state.inference_tx,
        |tx| InferenceRequest::Transcribe {
            request,
            expected_backend: Some("paraformer".to_string()),
            response_tx: tx,
        },
        TRANSCRIPTION_TIMEOUT,
    )
    .await?;

    res.render(Json(response));
    Ok(())
}

// ============================================================================
// Ascend backend endpoints
// ============================================================================

/// POST /v1/audio/asr/ascend — Qwen3-ASR on Ascend NPU
///
/// Routes directly to the Ascend backend (bypasses MLX inference thread).
#[handler]
pub async fn asr_ascend(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    if state.server_config.is_category_disabled("asr") {
        render_error(res, salvo::http::StatusCode::FORBIDDEN, "ASR is disabled", "forbidden");
        return Ok(());
    }

    let ascend_cfg = state.ascend_config.as_ref().ok_or_else(|| {
        tracing::error!("Ascend backend not configured");
        StatusError::internal_server_error()
    })?;

    if !ascend_cfg.has_asr() {
        render_error(res, salvo::http::StatusCode::SERVICE_UNAVAILABLE, "Ascend ASR not available", "unavailable");
        return Ok(());
    }

    let request: TranscriptionRequest = req
        .parse_json_with_max_size(10 * 1024 * 1024)
        .await
        .map_err(|e| {
            tracing::error!("Failed to parse request: {}", e);
            StatusError::bad_request()
        })?;

    // Run Ascend ASR in a blocking task (subprocess I/O)
    let cfg = ascend_cfg.clone();
    let response = tokio::task::spawn_blocking(move || {
        let engine = ascend::AscendAsrEngine::new((*cfg).clone())?;
        engine.transcribe(&request)
    })
    .await
    .map_err(|e| {
        tracing::error!("Ascend ASR task failed: {}", e);
        StatusError::internal_server_error()
    })?
    .map_err(|e| {
        tracing::error!("Ascend ASR error: {}", e);
        StatusError::internal_server_error()
    })?;

    res.render(Json(response));
    Ok(())
}

/// POST /v1/audio/tts/ascend — Qwen3-TTS on Ascend NPU (preset voices)
///
/// Routes directly to the Ascend backend.
#[handler]
pub async fn tts_ascend(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    if state.server_config.is_category_disabled("tts") {
        render_error(res, salvo::http::StatusCode::FORBIDDEN, "TTS is disabled", "forbidden");
        return Ok(());
    }

    let ascend_cfg = state.ascend_config.as_ref().ok_or_else(|| {
        tracing::error!("Ascend backend not configured");
        StatusError::internal_server_error()
    })?;

    if !ascend_cfg.has_tts() {
        render_error(res, salvo::http::StatusCode::SERVICE_UNAVAILABLE, "Ascend TTS not available", "unavailable");
        return Ok(());
    }

    // B3 §5.3.3: dispatch through the shared `TextToSpeech` trait instead of
    // constructing a fresh `AscendTtsEngine` per request. Backend selection
    // (subprocess vs FFI) is frozen at server startup via `ASCEND_TTS_TRANSPORT`.
    let backend = state.ascend_tts_backend.as_ref().ok_or_else(|| {
        tracing::error!("Ascend TTS backend not initialized");
        StatusError::internal_server_error()
    })?;

    let request: SpeechRequest = req.parse_json_with_max_size(10 * 1024 * 1024).await.map_err(|e| {
        tracing::error!("Failed to parse request: {}", e);
        StatusError::bad_request()
    })?;

    let tts_req = TtsTraitRequest {
        input: request.input,
        voice: request.voice.unwrap_or_else(|| "default".to_string()),
        language: request.language.unwrap_or_else(|| "English".to_string()),
        speed: request.speed,
        instruct: request.instruct,
    };

    let backend = backend.clone();
    let wav_data = tokio::task::spawn_blocking(move || backend.synthesize(tts_req))
        .await
        .map_err(|e| {
            tracing::error!("Ascend TTS task failed: {}", e);
            StatusError::internal_server_error()
        })?
        .map_err(|e| {
            tracing::error!("Ascend TTS error: {}", e);
            StatusError::internal_server_error()
        })?;

    let wav_bytes = match wav_data {
        TtsResponse::Wav(v) => v,
        // If a backend starts returning raw PCM in the future, the handler
        // can wrap it via `pcm_to_wav`; for v1 only WAV-returning backends
        // are in production.
        TtsResponse::Pcm { samples, sample_rate } => {
            let bytes: Vec<u8> = samples
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect();
            pcm_to_wav(&bytes, sample_rate)
        }
    };

    res.headers_mut()
        .insert("Content-Type", "audio/wav".parse().unwrap());
    res.write_body(wav_bytes).ok();
    Ok(())
}

/// POST /v1/audio/tts/ascend/clone — Qwen3-TTS voice cloning on Ascend NPU
///
/// Accepts multipart form with reference audio. Routes to Ascend backend.
#[handler]
pub async fn tts_ascend_clone(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    if state.server_config.is_category_disabled("tts") {
        render_error(res, salvo::http::StatusCode::FORBIDDEN, "TTS is disabled", "forbidden");
        return Ok(());
    }

    let ascend_cfg = state.ascend_config.as_ref().ok_or_else(|| {
        tracing::error!("Ascend backend not configured");
        StatusError::internal_server_error()
    })?;

    if !ascend_cfg.has_tts() {
        render_error(res, salvo::http::StatusCode::SERVICE_UNAVAILABLE, "Ascend TTS not available", "unavailable");
        return Ok(());
    }

    // B3 §5.3.3: dispatch through the shared `TextToSpeech` trait.
    let backend = state.ascend_tts_backend.as_ref().ok_or_else(|| {
        tracing::error!("Ascend TTS backend not initialized");
        StatusError::internal_server_error()
    })?;

    let request = parse_clone_multipart(req).await?;

    let tts_clone_req = TtsCloneRequest {
        input: request.input,
        reference_audio: request.reference_audio,
        language: request.language,
        speed: request.speed,
        instruct: request.instruct,
    };

    let backend = backend.clone();
    let wav_data = tokio::task::spawn_blocking(move || backend.synthesize_clone(tts_clone_req))
        .await
        .map_err(|e| {
            tracing::error!("Ascend TTS clone task failed: {}", e);
            StatusError::internal_server_error()
        })?
        .map_err(|e| {
            tracing::error!("Ascend TTS clone error: {}", e);
            StatusError::internal_server_error()
        })?;

    let wav_bytes = match wav_data {
        TtsResponse::Wav(v) => v,
        TtsResponse::Pcm { samples, sample_rate } => {
            let bytes: Vec<u8> = samples
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect();
            pcm_to_wav(&bytes, sample_rate)
        }
    };

    res.headers_mut()
        .insert("Content-Type", "audio/wav".parse().unwrap());
    res.write_body(wav_bytes).ok();
    Ok(())
}

/// POST /v1/audio/tts/outetts — OuteTTS on Ascend NPU
///
/// Alternative TTS using OuteTTS-0.2 model via llama-tts binary.
#[handler]
pub async fn tts_outetts(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    if state.server_config.is_category_disabled("tts") {
        render_error(res, salvo::http::StatusCode::FORBIDDEN, "TTS is disabled", "forbidden");
        return Ok(());
    }

    let ascend_cfg = state.ascend_config.as_ref().ok_or_else(|| {
        tracing::error!("Ascend backend not configured");
        StatusError::internal_server_error()
    })?;

    if !ascend_cfg.has_outetts() {
        render_error(res, salvo::http::StatusCode::SERVICE_UNAVAILABLE, "OuteTTS not available", "unavailable");
        return Ok(());
    }

    let request: SpeechRequest = req.parse_json_with_max_size(10 * 1024 * 1024).await.map_err(|e| {
        tracing::error!("Failed to parse request: {}", e);
        StatusError::bad_request()
    })?;

    let cfg = ascend_cfg.clone();
    let wav_data = tokio::task::spawn_blocking(move || {
        let engine = ascend::AscendOuteTtsEngine::new((*cfg).clone())?;
        engine.synthesize(&request.input)
    })
    .await
    .map_err(|e| {
        tracing::error!("OuteTTS task failed: {}", e);
        StatusError::internal_server_error()
    })?
    .map_err(|e| {
        tracing::error!("OuteTTS error: {}", e);
        StatusError::internal_server_error()
    })?;

    res.headers_mut()
        .insert("Content-Type", "audio/wav".parse().unwrap());
    res.write_body(wav_data).ok();
    Ok(())
}

// ============================================================================
// Per-sentence scheduling helpers
// ============================================================================

/// Spawn a task that submits each sentence as a separate queue item for preset TTS.
/// Returns a receiver that yields PCM chunks as they complete.
fn spawn_per_sentence_tts(
    inference_tx: mpsc::Sender<InferenceRequest>,
    sentences: Vec<String>,
    voice: String,
    language: String,
    speed: f32,
    instruct: Option<String>,
) -> mpsc::Receiver<AudioChunk> {
    let (chunk_tx, chunk_rx) = mpsc::channel::<AudioChunk>(32);
    tokio::spawn(async move {
        for sentence in &sentences {
            let (tx, rx) = tokio::sync::oneshot::channel();
            if inference_tx
                .send(InferenceRequest::Qwen3Tts(TtsRequest::SpeechOneSentence {
                    sentence: sentence.clone(),
                    voice: voice.clone(),
                    language: language.clone(),
                    speed,
                    instruct: instruct.clone(),
                    response_tx: tx,
                }))
                .await
                .is_err()
            {
                let _ = chunk_tx.send(AudioChunk::Error("Inference channel closed".into())).await;
                return;
            }
            match rx.await {
                Ok(Ok(pcm)) => {
                    if chunk_tx.send(AudioChunk::Pcm(pcm)).await.is_err() {
                        tracing::info!("Client disconnected during TTS streaming");
                        return;
                    }
                }
                Ok(Err(e)) => {
                    tracing::warn!("TTS sentence failed, skipping: {e}");
                    continue;
                }
                Err(_) => {
                    let _ = chunk_tx.send(AudioChunk::Error("Inference dropped".into())).await;
                    return;
                }
            }
        }
        let _ = chunk_tx
            .send(AudioChunk::Done { total_samples: 0, duration_secs: 0.0 })
            .await;
    });
    chunk_rx
}

/// Spawn a task that prepares clone ref then submits each sentence individually.
/// Returns a receiver that yields PCM chunks as they complete.
fn spawn_per_sentence_clone(
    inference_tx: mpsc::Sender<InferenceRequest>,
    sentences: Vec<String>,
    audio_bytes: Vec<u8>,
    language: String,
    speed: f32,
    instruct: Option<String>,
) -> mpsc::Receiver<AudioChunk> {
    let (chunk_tx, chunk_rx) = mpsc::channel::<AudioChunk>(32);
    tokio::spawn(async move {
        // Step 1: Prepare clone reference (load + resample + cache)
        let (tx, rx) = tokio::sync::oneshot::channel();
        if inference_tx
            .send(InferenceRequest::Qwen3Tts(TtsRequest::PrepareCloneRef {
                audio_bytes,
                response_tx: tx,
            }))
            .await
            .is_err()
        {
            let _ = chunk_tx.send(AudioChunk::Error("Inference channel closed".into())).await;
            return;
        }
        match rx.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                let _ = chunk_tx.send(AudioChunk::Error(e.to_string())).await;
                return;
            }
            Err(_) => {
                let _ = chunk_tx.send(AudioChunk::Error("Inference dropped".into())).await;
                return;
            }
        }

        // Step 2: Submit each sentence individually
        for sentence in &sentences {
            let (tx, rx) = tokio::sync::oneshot::channel();
            if inference_tx
                .send(InferenceRequest::Qwen3Tts(TtsRequest::CloneOneSentence {
                    sentence: sentence.clone(),
                    language: language.clone(),
                    speed,
                    instruct: instruct.clone(),
                    response_tx: tx,
                }))
                .await
                .is_err()
            {
                let _ = chunk_tx.send(AudioChunk::Error("Inference channel closed".into())).await;
                return;
            }
            match rx.await {
                Ok(Ok(pcm)) => {
                    if chunk_tx.send(AudioChunk::Pcm(pcm)).await.is_err() {
                        tracing::info!("Client disconnected during clone TTS streaming");
                        return;
                    }
                }
                Ok(Err(e)) => {
                    tracing::warn!("Clone TTS sentence failed, skipping: {e}");
                    continue;
                }
                Err(_) => {
                    let _ = chunk_tx.send(AudioChunk::Error("Inference dropped".into())).await;
                    return;
                }
            }
        }
        let _ = chunk_tx
            .send(AudioChunk::Done { total_samples: 0, duration_secs: 0.0 })
            .await;
    });
    chunk_rx
}

// ============================================================================
// Shared helpers
// ============================================================================

/// Parse a clone request from multipart form data.
///
/// Fields: `reference_audio` (file), `input` (text), `language` (text, optional),
/// `speed` (text, optional), `instruct`/`prompt` (text, optional).
async fn parse_clone_multipart(req: &mut Request) -> Result<SpeechCloneRequest, StatusError> {
    req.set_secure_max_size(10 * 1024 * 1024);
    let form = req.form_data().await.map_err(|e| {
        tracing::error!("Failed to parse multipart form: {e}");
        StatusError::bad_request()
    })?;

    let input = form
        .fields
        .get("input")
        .cloned()
        .ok_or_else(StatusError::bad_request)?;
    let language = form
        .fields
        .get("language")
        .cloned()
        .unwrap_or_else(|| "chinese".to_string());
    let speed: f32 = form
        .fields
        .get("speed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let instruct = form
        .fields
        .get("instruct")
        .cloned()
        .or_else(|| form.fields.get("prompt").cloned())
        .filter(|s| !s.trim().is_empty());
    let audio_path = form
        .files
        .get("reference_audio")
        .ok_or_else(StatusError::bad_request)?
        .path()
        .to_path_buf();

    let audio_bytes = tokio::fs::read(&audio_path).await.map_err(|e| {
        tracing::error!("Failed to read uploaded audio: {e}");
        StatusError::internal_server_error()
    })?;

    Ok(SpeechCloneRequest {
        input,
        reference_audio: audio_bytes,
        language,
        speed,
        instruct,
    })
}

/// Collect all PCM chunks from a streaming channel into a single buffer.
async fn collect_pcm_chunks(mut chunk_rx: mpsc::Receiver<AudioChunk>) -> Result<Vec<u8>, StatusError> {
    let mut pcm = Vec::new();
    while let Some(chunk) = chunk_rx.recv().await {
        match chunk {
            AudioChunk::Pcm(data) => pcm.extend_from_slice(&data),
            AudioChunk::Done { .. } => break,
            AudioChunk::Error(e) => {
                tracing::error!("TTS error during WAV collection: {e}");
                return Err(StatusError::internal_server_error());
            }
        }
    }
    Ok(pcm)
}

/// Wrap raw PCM bytes (16-bit signed LE, mono) in a WAV header.
fn pcm_to_wav(pcm: &[u8], sample_rate: u32) -> Vec<u8> {
    let data_len = pcm.len() as u32;
    let file_len = 36 + data_len;
    let mut wav = Vec::with_capacity(44 + pcm.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_len.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend_from_slice(pcm);
    wav
}

/// Set PCM streaming response headers and stream audio chunks.
fn stream_pcm_response(chunk_rx: mpsc::Receiver<AudioChunk>, res: &mut Response) {
    let stream = ReceiverStream::new(chunk_rx).filter_map(|chunk| match chunk {
        AudioChunk::Pcm(data) => {
            Some(Ok::<_, std::io::Error>(salvo::hyper::body::Bytes::from(data)))
        }
        AudioChunk::Done { .. } => None,
        AudioChunk::Error(e) => {
            tracing::error!("Streaming TTS error: {e}");
            None
        }
    });

    res.headers_mut()
        .insert("Content-Type", "audio/pcm".parse().unwrap());
    res.headers_mut()
        .insert("X-Audio-Sample-Rate", "24000".parse().unwrap());
    res.headers_mut()
        .insert("X-Audio-Channels", "1".parse().unwrap());
    res.headers_mut()
        .insert("X-Audio-Bits-Per-Sample", "16".parse().unwrap());
    res.headers_mut()
        .insert("Transfer-Encoding", "chunked".parse().unwrap());

    res.stream(stream);
}

#[cfg(test)]
mod tests {
    //! Handler-level tests for `POST /v1/audio/tts/qwen3` voice validation.
    //!
    //! These exercise the new 404 behavior without touching the real TTS
    //! pipeline. The 404 rejection short-circuits before any inference
    //! request is sent, so a dummy receiver is sufficient.
    //!
    //! For happy-path requests (registered voice) the test spawns a stub
    //! receiver that answers each `SpeechOneSentence` with a tiny PCM
    //! buffer, mirroring what the real inference thread would produce.
    use super::*;
    use crate::inference::TtsRequest as CoreTtsRequest;
    use crate::state::AppState;
    use salvo::test::{ResponseExt, TestClient};
    use serde_json::Value;
    use std::sync::Arc;
    use tokio::sync::{broadcast, mpsc};

    /// Build an `AppState` whose `inference_tx` is consumed by a stub
    /// that answers Qwen3 TTS sentence requests with a 4-byte PCM blob.
    /// Other channels drop their messages (they're not exercised by
    /// `tts_qwen3`).
    fn make_test_state() -> AppState {
        let (inference_tx, mut inference_rx) = mpsc::channel(8);
        tokio::spawn(async move {
            while let Some(req) = inference_rx.recv().await {
                if let crate::inference::InferenceRequest::Qwen3Tts(tts_req) = req {
                    match tts_req {
                        CoreTtsRequest::SpeechOneSentence { response_tx, .. } => {
                            let _ = response_tx.send(Ok(vec![0u8; 4]));
                        }
                        CoreTtsRequest::CloneOneSentence { response_tx, .. } => {
                            let _ = response_tx.send(Ok(vec![0u8; 4]));
                        }
                        CoreTtsRequest::PrepareCloneRef { response_tx, .. } => {
                            let _ = response_tx.send(Ok(()));
                        }
                        CoreTtsRequest::Speech { response_tx, .. } => {
                            let _ = response_tx.send(Ok(vec![0u8; 4]));
                        }
                        CoreTtsRequest::SpeechClone { response_tx, .. } => {
                            let _ = response_tx.send(Ok(vec![0u8; 4]));
                        }
                    }
                }
            }
        });
        let (training_tx, _training_rx) = mpsc::channel(1);
        let (progress_tx, _) = broadcast::channel(1);
        let (download_tx, _download_rx) = mpsc::channel(1);
        let (download_progress_tx, _) = broadcast::channel(1);
        AppState {
            inference_tx,
            training_tx,
            progress_tx,
            cancel_flag: Default::default(),
            download_tx,
            download_progress_tx,
            download_cancel_flags: Default::default(),
            server_config: Arc::new(crate::server_config::ServerConfig::default()),
            ascend_config: None,
            ascend_tts_backend: None,
        }
    }

    fn make_test_service() -> salvo::Service {
        let state = make_test_state();
        let router = Router::new().push(Router::with_path("v1/audio/tts/qwen3").post(tts_qwen3));
        let router = router.hoop(salvo::affix_state::inject(state));
        salvo::Service::new(router)
    }

    /// Point the registry at a scratch file so tests don't accidentally
    /// accept voices from the developer's real `~/.OminiX/models/voices.json`.
    /// Env mutation is serialized by `voice_test_guard`.
    fn set_voices_json_to_empty() -> tempfile::NamedTempFile {
        let file = tempfile::NamedTempFile::new().expect("tempfile");
        std::fs::write(file.path(), r#"{"voices":{}}"#).unwrap();
        std::env::set_var("OMINIX_VOICES_JSON", file.path());
        file
    }

    fn set_voices_json_with(content: &str) -> tempfile::NamedTempFile {
        let file = tempfile::NamedTempFile::new().expect("tempfile");
        std::fs::write(file.path(), content).unwrap();
        std::env::set_var("OMINIX_VOICES_JSON", file.path());
        file
    }

    /// Serialize tests that mutate `OMINIX_VOICES_JSON`. A tokio-aware
    /// mutex is used because the guard is held across await points
    /// within each test.
    async fn voice_test_guard() -> tokio::sync::MutexGuard<'static, ()> {
        use std::sync::OnceLock;
        use tokio::sync::Mutex;
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().await
    }

    #[tokio::test]
    async fn should_return_404_when_voice_is_unknown() {
        let _guard = voice_test_guard().await;
        let _temp = set_voices_json_to_empty();
        let service = make_test_service();

        let mut res = TestClient::post("http://127.0.0.1:5800/v1/audio/tts/qwen3")
            .json(&serde_json::json!({"voice": "yangmi", "input": "test"}))
            .send(&service)
            .await;

        assert_eq!(res.status_code.unwrap().as_u16(), 404);
        let body: Value = res.take_json().await.expect("json body");
        assert_eq!(body["error"], "voice_not_found");
        assert_eq!(body["requested_voice"], "yangmi");
        assert!(
            body["available_voices"]
                .as_array()
                .expect("available_voices array")
                .iter()
                .any(|v| v == "vivian"),
            "expected vivian in available_voices: {body}"
        );
        let msg = body["message"].as_str().unwrap();
        assert!(msg.contains("yangmi"));
        assert!(msg.contains("vivian"));
    }

    #[tokio::test]
    async fn should_return_200_when_voice_is_registered_preset() {
        let _guard = voice_test_guard().await;
        let _temp = set_voices_json_to_empty();
        let service = make_test_service();

        // Ask for WAV so the handler awaits the stub receiver's PCM and
        // returns a complete body rather than streaming chunked.
        let res = TestClient::post("http://127.0.0.1:5800/v1/audio/tts/qwen3?format=wav")
            .json(&serde_json::json!({"voice": "vivian", "input": "hello"}))
            .send(&service)
            .await;

        assert_eq!(res.status_code.unwrap().as_u16(), 200);
    }

    #[tokio::test]
    async fn should_return_200_when_voice_field_is_missing() {
        let _guard = voice_test_guard().await;
        let _temp = set_voices_json_to_empty();
        let service = make_test_service();

        let res = TestClient::post("http://127.0.0.1:5800/v1/audio/tts/qwen3?format=wav")
            .json(&serde_json::json!({"input": "hello"}))
            .send(&service)
            .await;

        // Empty voice normalizes to "vivian" which is a preset — happy path.
        assert_eq!(res.status_code.unwrap().as_u16(), 200);
    }

    #[tokio::test]
    async fn should_return_200_when_voice_matches_custom_alias() {
        let _guard = voice_test_guard().await;
        let _temp = set_voices_json_with(r#"{"voices":{"mia_clone":{"aliases":["mia"]}}}"#);
        let service = make_test_service();

        let res = TestClient::post("http://127.0.0.1:5800/v1/audio/tts/qwen3?format=wav")
            .json(&serde_json::json!({"voice": "mia", "input": "hi"}))
            .send(&service)
            .await;

        assert_eq!(res.status_code.unwrap().as_u16(), 200);
    }

    #[test]
    fn should_include_requested_voice_when_error_serializes() {
        // Sanity-check the VoiceNotFoundError shape matches the contract.
        let err = VoiceNotFoundError {
            error: "voice_not_found",
            message: "Voice 'x' is not registered. Available: vivian".into(),
            requested_voice: "x".into(),
            available_voices: vec!["vivian".into()],
        };
        let v: Value = serde_json::from_str(&serde_json::to_string(&err).unwrap()).unwrap();
        assert_eq!(v["error"], "voice_not_found");
        assert_eq!(v["requested_voice"], "x");
        assert_eq!(v["available_voices"][0], "vivian");
        assert!(v["message"].as_str().unwrap().contains("vivian"));
    }
}
