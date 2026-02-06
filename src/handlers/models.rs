use std::time::Duration;

use salvo::prelude::*;
use tokio::sync::oneshot;
use tokio::time::timeout;

use crate::error::render_error;
use crate::inference::InferenceRequest;

use super::helpers::get_state;

/// Timeout for model loading
const MODEL_LOAD_TIMEOUT: Duration = Duration::from_secs(300); // 5 minutes

/// POST /v1/models/load - Load a model dynamically
///
/// Request body:
/// - model: Model ID or path (required)
/// - model_type: "llm", "asr", "tts", or "image" (default: "llm")
#[handler]
pub async fn load_model(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    #[derive(serde::Deserialize)]
    struct LoadModelRequest {
        model: String,
        #[serde(default = "default_model_type")]
        model_type: String,
    }

    fn default_model_type() -> String {
        "llm".to_string()
    }

    let request: LoadModelRequest = req.parse_json().await.map_err(|e| {
        tracing::error!("Failed to parse request: {}", e);
        StatusError::bad_request()
    })?;

    let model_type = request.model_type.as_str();
    tracing::info!("Loading model: {} (type: {})", request.model, model_type);

    // Validate model paths to prevent directory traversal
    if !crate::utils::is_safe_path(&request.model) {
        render_error(
            res,
            salvo::http::StatusCode::BAD_REQUEST,
            "Model path contains invalid directory traversal",
            "invalid_request_error",
        );
        return Ok(());
    }

    let inference_request = match model_type {
        "llm" => {
            let (response_tx, response_rx) = oneshot::channel();
            state
                .inference_tx
                .send(InferenceRequest::LoadLlmModel {
                    model_id: request.model.clone(),
                    response_tx,
                })
                .await
                .map_err(|_| StatusError::internal_server_error())?;
            response_rx
        }
        "asr" => {
            let (response_tx, response_rx) = oneshot::channel();
            state
                .inference_tx
                .send(InferenceRequest::LoadAsrModel {
                    model_dir: request.model.clone(),
                    response_tx,
                })
                .await
                .map_err(|_| StatusError::internal_server_error())?;
            response_rx
        }
        "tts" => {
            let (response_tx, response_rx) = oneshot::channel();
            state
                .inference_tx
                .send(InferenceRequest::LoadTtsModel {
                    ref_audio: request.model.clone(),
                    response_tx,
                })
                .await
                .map_err(|_| StatusError::internal_server_error())?;
            response_rx
        }
        "image" => {
            let (response_tx, response_rx) = oneshot::channel();
            state
                .inference_tx
                .send(InferenceRequest::LoadImageModel {
                    model_id: request.model.clone(),
                    response_tx,
                })
                .await
                .map_err(|_| StatusError::internal_server_error())?;
            response_rx
        }
        _ => {
            render_error(
                res,
                salvo::http::StatusCode::BAD_REQUEST,
                &format!(
                    "Unknown model_type: {}. Use: llm, asr, tts, or image",
                    model_type
                ),
                "invalid_request_error",
            );
            return Ok(());
        }
    };

    let result = timeout(MODEL_LOAD_TIMEOUT, inference_request)
        .await
        .map_err(|_| {
            tracing::error!("Model loading timed out after {:?}", MODEL_LOAD_TIMEOUT);
            StatusError::gateway_timeout()
        })?
        .map_err(|_| StatusError::internal_server_error())?;

    match result {
        Ok(loaded_model) => {
            res.render(Json(serde_json::json!({
                "status": "success",
                "model": loaded_model,
                "model_type": model_type
            })));
        }
        Err(e) => {
            render_error(
                res,
                salvo::http::StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(),
                "server_error",
            );
        }
    }

    Ok(())
}

/// POST /v1/models/unload - Unload a model to free memory
///
/// Request body:
/// - model_type: "llm", "asr", "tts", "image", or "all"
#[handler]
pub async fn unload_model(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    #[derive(serde::Deserialize)]
    struct UnloadModelRequest {
        model_type: String,
    }

    let request: UnloadModelRequest = req.parse_json().await.map_err(|e| {
        tracing::error!("Failed to parse request: {}", e);
        StatusError::bad_request()
    })?;

    tracing::info!("Unloading model type: {}", request.model_type);

    let (response_tx, response_rx) = oneshot::channel();
    state
        .inference_tx
        .send(InferenceRequest::UnloadModel {
            model_type: request.model_type.clone(),
            response_tx,
        })
        .await
        .map_err(|_| StatusError::internal_server_error())?;

    let result = timeout(Duration::from_secs(30), response_rx)
        .await
        .map_err(|_| {
            tracing::error!("Model unloading timed out");
            StatusError::gateway_timeout()
        })?
        .map_err(|_| StatusError::internal_server_error())?;

    match result {
        Ok(message) => {
            res.render(Json(serde_json::json!({
                "status": "success",
                "message": message
            })));
        }
        Err(e) => {
            render_error(
                res,
                salvo::http::StatusCode::BAD_REQUEST,
                &e.to_string(),
                "invalid_request_error",
            );
        }
    }

    Ok(())
}
