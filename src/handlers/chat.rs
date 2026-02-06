use std::time::Duration;

use salvo::prelude::*;

use crate::inference::InferenceRequest;
use crate::types::ChatCompletionRequest;

use super::helpers::{get_state, send_and_wait};

/// Timeout for LLM chat completions (can be long for large responses)
const CHAT_TIMEOUT: Duration = Duration::from_secs(300); // 5 minutes

/// POST /v1/chat/completions - OpenAI-compatible chat completions
#[handler]
pub async fn chat_completions(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    let request: ChatCompletionRequest = req.parse_json().await.map_err(|e| {
        tracing::error!("Failed to parse request: {}", e);
        StatusError::bad_request()
    })?;

    if request.stream.unwrap_or(false) {
        tracing::warn!("Streaming not yet implemented, falling back to non-streaming");
    }

    let response = send_and_wait(
        &state.inference_tx,
        |tx| InferenceRequest::Chat { request, response_tx: tx },
        CHAT_TIMEOUT,
    )
    .await?;

    res.render(Json(response));
    Ok(())
}
