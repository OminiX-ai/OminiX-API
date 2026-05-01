use std::time::Duration;

use salvo::prelude::*;

use crate::inference::InferenceRequest;
use crate::types::VideoGenerationRequest;

use super::helpers::{get_state, send_and_wait};

const VIDEO_TIMEOUT: Duration = Duration::from_secs(1800); // 30 minutes

/// POST /v1/videos/generations - Video generation
#[handler]
pub async fn videos_generations(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    let request: VideoGenerationRequest = req
        .parse_json_with_max_size(1024 * 1024)
        .await
        .map_err(|e| {
            tracing::error!("Failed to parse video request: {}", e);
            StatusError::bad_request()
        })?;

    let response = send_and_wait(
        &state.inference_tx,
        |tx| InferenceRequest::Video { request, response_tx: tx },
        VIDEO_TIMEOUT,
    )
    .await?;

    res.render(Json(response));
    Ok(())
}
