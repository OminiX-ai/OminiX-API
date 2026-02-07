use salvo::prelude::*;
use tokio::sync::oneshot;

use crate::inference::{InferenceRequest, ModelStatus};

use super::helpers::get_state;

/// GET /health - Health check
#[handler]
pub async fn health(res: &mut Response) {
    res.render(Json(serde_json::json!({
        "status": "healthy",
        "service": "ominix-api"
    })));
}

/// GET /v1/models - List available models (queries live status)
#[handler]
pub async fn list_models(depot: &mut Depot, res: &mut Response) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    let (response_tx, response_rx) = oneshot::channel();
    state
        .inference_tx
        .send(InferenceRequest::GetModelStatus { response_tx })
        .await
        .map_err(|_| StatusError::internal_server_error())?;

    let status: ModelStatus = response_rx
        .await
        .map_err(|_| StatusError::internal_server_error())?;

    let now = chrono::Utc::now().timestamp();
    let mut data = Vec::new();

    if let Some(ref id) = status.llm {
        data.push(serde_json::json!({
            "id": id, "object": "model", "created": now,
            "owned_by": "local", "type": "llm"
        }));
    }
    if let Some(ref id) = status.asr {
        data.push(serde_json::json!({
            "id": "paraformer", "object": "model", "created": now,
            "owned_by": "local", "type": "asr"
        }));
        let _ = id; // asr stores path, not display name
    }
    if let Some(ref _id) = status.tts {
        data.push(serde_json::json!({
            "id": "gpt-sovits", "object": "model", "created": now,
            "owned_by": "local", "type": "tts"
        }));
    }
    if let Some(ref id) = status.image {
        data.push(serde_json::json!({
            "id": id, "object": "model", "created": now,
            "owned_by": "local", "type": "image"
        }));
    }
    if let Some(ref id) = status.vlm {
        data.push(serde_json::json!({
            "id": id, "object": "model", "created": now,
            "owned_by": "local", "type": "vlm"
        }));
    }

    res.render(Json(serde_json::json!({
        "object": "list",
        "data": data
    })));
    Ok(())
}

/// GET /v1/models/report - Model availability report
#[handler]
pub async fn models_report(res: &mut Response) {
    let report = crate::model_config::get_model_report();
    res.render(Json(report));
}

/// GET /v1/models/status - Get current model status
#[handler]
pub async fn model_status(depot: &mut Depot, res: &mut Response) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    let (response_tx, response_rx) = oneshot::channel();
    state
        .inference_tx
        .send(InferenceRequest::GetModelStatus { response_tx })
        .await
        .map_err(|_| StatusError::internal_server_error())?;

    let status: ModelStatus = response_rx
        .await
        .map_err(|_| StatusError::internal_server_error())?;

    res.render(Json(serde_json::json!({
        "status": "success",
        "models": status
    })));
    Ok(())
}
