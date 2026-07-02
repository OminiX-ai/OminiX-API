use std::time::Duration;

use salvo::prelude::*;
use tokio::sync::oneshot;
use tokio::time::timeout;

use crate::error::render_error;
use crate::inference::InferenceRequest;
use crate::types::VideoGenerationRequest;

use super::helpers::get_state;

const VIDEO_TIMEOUT: Duration = Duration::from_secs(3600); // Wan2.2 I2V can run well past 15 minutes.

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

    let (response_tx, response_rx) = oneshot::channel();
    state
        .inference_tx
        .send(InferenceRequest::Video { request, response_tx })
        .await
        .map_err(|_| StatusError::internal_server_error())?;

    let response = match timeout(VIDEO_TIMEOUT, response_rx).await {
        Err(_) => {
            render_error(
                res,
                salvo::http::StatusCode::GATEWAY_TIMEOUT,
                "Video generation timed out.",
                "timeout",
            );
            return Ok(());
        }
        Ok(Err(_)) => {
            render_error(
                res,
                salvo::http::StatusCode::INTERNAL_SERVER_ERROR,
                "Video generation worker stopped before returning a result.",
                "internal_error",
            );
            return Ok(());
        }
        Ok(Ok(Err(e))) => {
            let message = format!("{:#}", e);
            tracing::error!("Video inference error: {}", message);
            render_error(
                res,
                salvo::http::StatusCode::INTERNAL_SERVER_ERROR,
                &message,
                "inference_error",
            );
            return Ok(());
        }
        Ok(Ok(Ok(response))) => response,
    };

    res.render(Json(response));
    Ok(())
}
