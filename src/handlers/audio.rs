use std::time::Duration;

use salvo::prelude::*;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::error::render_error;
use crate::inference::{AudioChunk, InferenceRequest, TtsRequest};
use crate::types::{SpeechCloneRequest, SpeechRequest, TranscriptionRequest};

use super::helpers::{get_state, send_and_wait};

/// Timeout for audio transcription
const TRANSCRIPTION_TIMEOUT: Duration = Duration::from_secs(1800); // 30 minutes
/// Timeout for text-to-speech (non-streaming WAV fallback)
const TTS_TIMEOUT: Duration = Duration::from_secs(300); // 5 minutes (pool handles long text)
/// Timeout for voice cloning (loading ref audio + synthesis)
const TTS_CLONE_TIMEOUT: Duration = Duration::from_secs(300); // 5 minutes

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

    // Always use sentence-level streaming to avoid GPU OOM on long text.
    let wants_wav = req.query::<String>("format").as_deref() == Some("wav")
        || request.response_format == "wav"
        || request.reference_audio.is_some();

    let (chunk_tx, chunk_rx) = mpsc::channel::<AudioChunk>(32);

    state
        .inference_tx
        .send(InferenceRequest::Qwen3Tts(TtsRequest::SpeechStream {
            request,
            chunk_tx,
        }))
        .await
        .map_err(|_| StatusError::internal_server_error())?;

    if wants_wav {
        let pcm_data = collect_pcm_chunks(chunk_rx).await?;
        let wav_data = pcm_to_wav(&pcm_data, 24000);
        res.headers_mut()
            .insert("Content-Type", "audio/wav".parse().unwrap());
        res.write_body(wav_data).ok();
        return Ok(());
    }

    // Stream PCM chunks back to client
    let stream = ReceiverStream::new(chunk_rx).filter_map(|chunk| match chunk {
        AudioChunk::Pcm(data) => Some(Ok::<_, std::io::Error>(salvo::hyper::body::Bytes::from(data))),
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

    // Always use sentence-level streaming to avoid GPU OOM on long text.
    let (chunk_tx, chunk_rx) = mpsc::channel::<AudioChunk>(32);

    state
        .inference_tx
        .send(InferenceRequest::Qwen3Tts(TtsRequest::SpeechCloneStream {
            request,
            chunk_tx,
        }))
        .await
        .map_err(|_| StatusError::internal_server_error())?;

    if wants_wav {
        let pcm_data = collect_pcm_chunks(chunk_rx).await?;
        let wav_data = pcm_to_wav(&pcm_data, 24000);
        res.headers_mut()
            .insert("Content-Type", "audio/wav".parse().unwrap());
        res.write_body(wav_data).ok();
        return Ok(());
    }

    let stream = ReceiverStream::new(chunk_rx).filter_map(|chunk| match chunk {
        AudioChunk::Pcm(data) => Some(Ok::<_, std::io::Error>(salvo::hyper::body::Bytes::from(data))),
        AudioChunk::Done { .. } => None,
        AudioChunk::Error(e) => {
            tracing::error!("Streaming clone TTS error: {e}");
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

    // Always use sentence-level streaming to avoid GPU OOM on long text.
    // For WAV format, we collect all streamed chunks and wrap in a WAV header.
    let (chunk_tx, chunk_rx) = mpsc::channel::<AudioChunk>(32);

    state
        .inference_tx
        .send(InferenceRequest::Qwen3Tts(TtsRequest::SpeechStream { request, chunk_tx }))
        .await
        .map_err(|_| StatusError::internal_server_error())?;

    if wants_wav {
        // Collect streamed PCM chunks into a complete WAV response
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

    // Always use sentence-level streaming to avoid GPU OOM on long text.
    let (chunk_tx, chunk_rx) = mpsc::channel::<AudioChunk>(32);

    state
        .inference_tx
        .send(InferenceRequest::Qwen3Tts(TtsRequest::SpeechCloneStream { request, chunk_tx }))
        .await
        .map_err(|_| StatusError::internal_server_error())?;

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
