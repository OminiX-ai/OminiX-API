use std::time::Duration;

use salvo::prelude::*;

use crate::engines::ascend;
use crate::inference::InferenceRequest;
use crate::types::VlmCompletionRequest;

use super::helpers::{get_state, send_and_wait};

/// Timeout for VLM completions
const VLM_TIMEOUT: Duration = Duration::from_secs(300); // 5 minutes

/// POST /v1/vlm/completions - VLM image understanding
#[handler]
pub async fn vlm_completions(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    // 10MB limit for base64 image
    let request: VlmCompletionRequest = req
        .parse_json_with_max_size(10 * 1024 * 1024)
        .await
        .map_err(|e| {
            tracing::error!("Failed to parse VLM request: {}", e);
            StatusError::bad_request()
        })?;

    let response = send_and_wait(
        &state.inference_tx,
        |tx| InferenceRequest::VlmCompletion {
            request,
            response_tx: tx,
        },
        VLM_TIMEOUT,
    )
    .await?;

    res.render(Json(response));
    Ok(())
}

/// POST /v1/vlm/completions/ascend — VLM on Ascend NPU
///
/// Proxies to a llama-server instance with multimodal projector.
#[handler]
pub async fn vlm_completions_ascend(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    let ascend_cfg = state.ascend_config.as_ref().ok_or_else(|| {
        tracing::error!("Ascend backend not configured");
        StatusError::internal_server_error()
    })?;

    if !ascend_cfg.has_vlm() {
        crate::error::render_error(
            res,
            salvo::http::StatusCode::SERVICE_UNAVAILABLE,
            "Ascend VLM not available. Set ASCEND_VLM_MODEL and ASCEND_VLM_MMPROJ.",
            "unavailable",
        );
        return Ok(());
    }

    let request: VlmCompletionRequest = req
        .parse_json_with_max_size(10 * 1024 * 1024)
        .await
        .map_err(|e| {
            tracing::error!("Failed to parse VLM request: {}", e);
            StatusError::bad_request()
        })?;

    let cfg = ascend_cfg.clone();
    let response = tokio::task::spawn_blocking(move || {
        let server = ascend::AscendVlmServer::new((*cfg).clone())?;
        server.vlm_completion(&request)
    })
    .await
    .map_err(|e| {
        tracing::error!("Ascend VLM task failed: {}", e);
        StatusError::internal_server_error()
    })?
    .map_err(|e| {
        tracing::error!("Ascend VLM error: {}", e);
        StatusError::internal_server_error()
    })?;

    res.render(Json(response));
    Ok(())
}
