use std::time::Duration;

use salvo::prelude::*;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::inference::{AudioChunk, InferenceRequest, TtsRequest};
use crate::types::{SpeechCloneRequest, SpeechRequest, TranscriptionRequest};

use super::helpers::{get_state, send_and_wait, send_tts_and_wait};

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
        |tx| InferenceRequest::Transcribe { request, response_tx: tx },
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

    let request: SpeechRequest = req.parse_json().await.map_err(|e| {
        tracing::error!("Failed to parse request: {}", e);
        StatusError::bad_request()
    })?;

    // Check if client wants WAV format (non-streaming) or has reference_audio (clone)
    let wants_wav = req.query::<String>("format").as_deref() == Some("wav")
        || request.response_format == "wav"
        || request.reference_audio.is_some();

    if wants_wav {
        // Non-streaming: full WAV response → routed to TTS pool
        let audio_data = send_tts_and_wait(
            &state.tts_pool_tx,
            |tx| TtsRequest::Speech { request, response_tx: tx },
            TTS_TIMEOUT,
        )
        .await?;

        res.headers_mut()
            .insert("Content-Type", "audio/wav".parse().unwrap());
        res.write_body(audio_data).ok();
        return Ok(());
    }

    // Streaming: send PCM chunks as they're generated → routed to TTS pool
    let (chunk_tx, chunk_rx) = mpsc::channel::<AudioChunk>(32);

    state
        .tts_pool_tx
        .send(TtsRequest::SpeechStream {
            request,
            chunk_tx,
        })
        .await
        .map_err(|_| StatusError::internal_server_error())?;

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
/// Dedicated endpoint for voice cloning with reference audio.
/// Routed to TTS pool which auto-loads Base model with ECAPA-TDNN speaker encoder.
#[handler]
pub async fn audio_speech_clone(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    let request: SpeechCloneRequest = req.parse_json().await.map_err(|e| {
        tracing::error!("Failed to parse clone request: {}", e);
        StatusError::bad_request()
    })?;

    let audio_data = send_tts_and_wait(
        &state.tts_pool_tx,
        |tx| TtsRequest::SpeechClone { request, response_tx: tx },
        TTS_CLONE_TIMEOUT,
    )
    .await?;

    res.headers_mut()
        .insert("Content-Type", "audio/wav".parse().unwrap());
    res.write_body(audio_data).ok();
    Ok(())
}
