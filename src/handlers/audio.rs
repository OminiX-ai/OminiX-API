use std::time::Duration;

use salvo::prelude::*;

use crate::inference::InferenceRequest;
use crate::types::{SpeechCloneRequest, SpeechRequest, TranscriptionRequest};

use super::helpers::{get_state, send_and_wait};

/// Timeout for audio transcription
const TRANSCRIPTION_TIMEOUT: Duration = Duration::from_secs(1800); // 30 minutes
/// Timeout for text-to-speech (preset voices)
const TTS_TIMEOUT: Duration = Duration::from_secs(60); // 1 minute
/// Timeout for voice cloning (loading ref audio + synthesis)
const TTS_CLONE_TIMEOUT: Duration = Duration::from_secs(120); // 2 minutes

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

/// POST /v1/audio/speech - Text-to-speech
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

/// POST /v1/audio/speech/clone - Voice cloning TTS
///
/// Dedicated endpoint for voice cloning with reference audio.
/// Always uses the Base model (with ECAPA-TDNN speaker encoder).
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

    let audio_data = send_and_wait(
        &state.inference_tx,
        |tx| InferenceRequest::SpeechClone { request, response_tx: tx },
        TTS_CLONE_TIMEOUT,
    )
    .await?;

    res.headers_mut()
        .insert("Content-Type", "audio/wav".parse().unwrap());
    res.write_body(audio_data).ok();
    Ok(())
}
