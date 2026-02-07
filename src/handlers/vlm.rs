use std::time::Duration;

use salvo::prelude::*;

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
