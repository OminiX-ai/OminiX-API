use std::time::Duration;

use salvo::prelude::*;
use salvo::sse::{SseEvent, SseKeepAlive};
use tokio::time::timeout;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

use crate::error::render_error;
use crate::types::*;

use super::helpers::get_state;

/// GET /v1/voices - List all registered voices
#[handler]
pub async fn list_voices(res: &mut Response) {
    let voices_path = crate::utils::expand_tilde("~/.OminiX/models/voices.json");
    let voices = match std::fs::read_to_string(&voices_path) {
        Ok(content) => match serde_json::from_str::<serde_json::Value>(&content) {
            Ok(config) => {
                let mut voice_list = Vec::new();
                if let Some(voices) = config.get("voices").and_then(|v| v.as_object()) {
                    for (name, voice) in voices {
                        let aliases = voice
                            .get("aliases")
                            .and_then(|a| a.as_array())
                            .map(|a| {
                                a.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();
                        voice_list.push(VoiceInfo {
                            name: name.clone(),
                            aliases,
                        });
                    }
                }
                voice_list
            }
            Err(_) => Vec::new(),
        },
        Err(_) => Vec::new(),
    };

    res.render(Json(VoiceListResponse { voices }));
}

/// POST /v1/voices/train - Start voice cloning training
#[handler]
pub async fn start_voice_training(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    // Parse request (50MB limit for audio data)
    let request: VoiceTrainRequest = req
        .parse_json_with_max_size(50 * 1024 * 1024)
        .await
        .map_err(|e| {
            tracing::error!("Failed to parse training request: {}", e);
            StatusError::bad_request()
        })?;

    // Validate and sanitize voice name to prevent path traversal
    let voice_name = match crate::utils::sanitize_voice_name(&request.voice_name) {
        Ok(name) => name,
        Err(e) => {
            render_error(
                res,
                salvo::http::StatusCode::BAD_REQUEST,
                &e.to_string(),
                "invalid_request_error",
            );
            return Ok(());
        }
    };

    if request.transcript.is_empty() {
        render_error(
            res,
            salvo::http::StatusCode::BAD_REQUEST,
            "transcript is required",
            "invalid_request_error",
        );
        return Ok(());
    }

    // Generate task ID
    let task_id = format!("train-{}", uuid::Uuid::new_v4().simple());

    // Decode and save audio to a temporary directory
    let base_dir = std::env::var("TRAINING_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .map(|h| h.join(".OminiX/training"))
                .unwrap_or_else(|| std::path::PathBuf::from("/tmp/ominix-training"))
        });
    let work_dir = base_dir.join(&task_id);
    std::fs::create_dir_all(&work_dir).map_err(|_| StatusError::internal_server_error())?;

    let audio_bytes = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        &request.audio,
    )
    .map_err(|e| {
        tracing::error!("Base64 decode failed: {}", e);
        StatusError::bad_request()
    })?;

    let audio_path = work_dir.join("ref_audio.wav");
    std::fs::write(&audio_path, &audio_bytes).map_err(|_| StatusError::internal_server_error())?;

    // Send training request to training thread
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    state
        .training_tx
        .send(crate::training::TrainingRequest::StartTraining {
            task_id: task_id.clone(),
            voice_name: voice_name.clone(),
            audio_path,
            transcript: request.transcript,
            quality: request.quality,
            language: request.language,
            denoise: request.denoise,
            response_tx,
        })
        .await
        .map_err(|_| StatusError::internal_server_error())?;

    // Wait for acknowledgement (not completion)
    match response_rx.await {
        Ok(Ok(())) => {
            res.render(Json(VoiceTrainResponse {
                task_id,
                status: "accepted".to_string(),
                message: format!("Training started for voice '{}'", voice_name),
            }));
        }
        Ok(Err(e)) => {
            render_error(
                res,
                salvo::http::StatusCode::CONFLICT,
                &e.to_string(),
                "server_error",
            );
        }
        Err(_) => {
            render_error(
                res,
                salvo::http::StatusCode::SERVICE_UNAVAILABLE,
                "Training thread unavailable",
                "server_error",
            );
        }
    }

    Ok(())
}

/// GET /v1/voices/train/status - Get training task status
#[handler]
pub async fn get_training_status(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    let task_id: String = req.query::<String>("task_id").unwrap_or_default();
    if task_id.is_empty() {
        render_error(
            res,
            salvo::http::StatusCode::BAD_REQUEST,
            "task_id query parameter required",
            "invalid_request_error",
        );
        return Ok(());
    }

    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    state
        .training_tx
        .send(crate::training::TrainingRequest::GetStatus {
            task_id: task_id.clone(),
            response_tx,
        })
        .await
        .map_err(|_| StatusError::internal_server_error())?;

    match timeout(Duration::from_secs(5), response_rx).await {
        Ok(Ok(Some(status))) => {
            res.render(Json(status));
        }
        Ok(Ok(None)) => {
            render_error(
                res,
                salvo::http::StatusCode::NOT_FOUND,
                &format!("Task not found: {}", task_id),
                "not_found_error",
            );
        }
        _ => {
            render_error(
                res,
                salvo::http::StatusCode::SERVICE_UNAVAILABLE,
                "Training thread unavailable",
                "server_error",
            );
        }
    }

    Ok(())
}

/// GET /v1/voices/train/progress - SSE streaming training progress
#[handler]
pub async fn training_progress_sse(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    let task_id: String = req.query::<String>("task_id").unwrap_or_default();
    if task_id.is_empty() {
        render_error(
            res,
            salvo::http::StatusCode::BAD_REQUEST,
            "task_id query parameter required",
            "invalid_request_error",
        );
        return Ok(());
    }

    // Subscribe to progress events
    let rx = state.progress_tx.subscribe();

    // Filter for this task's events and convert to SSE
    let task_id_clone = task_id.clone();
    let stream = BroadcastStream::new(rx).filter_map(move |result| match result {
        Ok(event) if event.task_id == task_id_clone => {
            let data = serde_json::to_string(&event).ok()?;
            let sse_event = SseEvent::default().text(data);
            Some(Ok::<_, std::convert::Infallible>(sse_event))
        }
        _ => None,
    });

    SseKeepAlive::new(stream).stream(res);
    Ok(())
}

/// POST /v1/voices/train/cancel - Cancel training
#[handler]
pub async fn cancel_training(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    let task_id: String = req.query::<String>("task_id").unwrap_or_default();
    if task_id.is_empty() {
        render_error(
            res,
            salvo::http::StatusCode::BAD_REQUEST,
            "task_id query parameter required",
            "invalid_request_error",
        );
        return Ok(());
    }

    match crate::training::cancel_training_task(&state.cancel_flag, &task_id) {
        Ok(()) => {
            res.render(Json(serde_json::json!({
                "status": "success",
                "message": format!("Training {} cancelled", task_id)
            })));
        }
        Err(e) => {
            render_error(
                res,
                salvo::http::StatusCode::NOT_FOUND,
                &e.to_string(),
                "not_found_error",
            );
        }
    }

    Ok(())
}
