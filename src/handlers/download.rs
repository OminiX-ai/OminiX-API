use salvo::prelude::*;
use salvo::sse::{SseEvent, SseKeepAlive};
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

use crate::error::render_error;
use crate::types::*;

use super::helpers::get_state;

/// GET /v1/models/catalog - Full model catalog with download status
#[handler]
pub async fn model_catalog(res: &mut Response) {
    let catalog = crate::model_registry::get_model_catalog();
    res.render(Json(serde_json::json!({
        "models": catalog,
        "total": catalog.len(),
    })));
}

/// POST /v1/models/download - Start downloading a model
#[handler]
pub async fn download_model(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    let request: DownloadModelRequest = req.parse_json().await.map_err(|e| {
        tracing::error!("Failed to parse download request: {}", e);
        StatusError::bad_request()
    })?;

    // Look up the model spec
    let spec = match crate::model_registry::get_download_spec(&request.model_id) {
        Some(s) => s,
        None => {
            render_error(
                res,
                salvo::http::StatusCode::NOT_FOUND,
                &format!("Model '{}' not found in catalog", request.model_id),
                "not_found_error",
            );
            return Ok(());
        }
    };

    // Check if already downloaded
    let availability = crate::model_config::check_model(&request.model_id, spec.category.clone());
    if matches!(availability, crate::model_config::ModelAvailability::Ready { .. }) {
        render_error(
            res,
            salvo::http::StatusCode::CONFLICT,
            &format!("Model '{}' is already downloaded", request.model_id),
            "conflict_error",
        );
        return Ok(());
    }

    // Check if download is already in progress
    {
        let flags = state.download_cancel_flags.lock().unwrap();
        if flags.contains_key(&request.model_id) {
            render_error(
                res,
                salvo::http::StatusCode::CONFLICT,
                &format!("Download already in progress for '{}'", request.model_id),
                "conflict_error",
            );
            return Ok(());
        }
    }

    let task_id = format!("dl-{}", uuid::Uuid::new_v4().simple());
    let model_id = request.model_id.clone();

    // Send download request to download thread
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    state
        .download_tx
        .send(crate::download::DownloadRequest::StartDownload {
            task_id: task_id.clone(),
            spec,
            response_tx,
        })
        .await
        .map_err(|_| StatusError::internal_server_error())?;

    // Wait for acknowledgement
    match response_rx.await {
        Ok(Ok(())) => {
            res.render(Json(DownloadModelResponse {
                task_id,
                model_id,
                status: "accepted".to_string(),
                message: "Download started".to_string(),
            }));
        }
        Ok(Err(e)) => {
            render_error(
                res,
                salvo::http::StatusCode::CONFLICT,
                &e,
                "server_error",
            );
        }
        Err(_) => {
            render_error(
                res,
                salvo::http::StatusCode::SERVICE_UNAVAILABLE,
                "Download thread unavailable",
                "server_error",
            );
        }
    }

    Ok(())
}

/// GET /v1/models/download/progress - SSE streaming download progress
#[handler]
pub async fn download_progress_sse(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    let model_id: String = req.query::<String>("model_id").unwrap_or_default();

    // Subscribe to progress events
    let rx = state.download_progress_tx.subscribe();

    // Filter for this model's events (or all if no model_id specified) and convert to SSE
    let stream = BroadcastStream::new(rx).filter_map(move |result| match result {
        Ok(event) => {
            if !model_id.is_empty() && event.model_id != model_id {
                return None;
            }
            let data = serde_json::to_string(&event).ok()?;
            let sse_event = SseEvent::default().text(data);
            Some(Ok::<_, std::convert::Infallible>(sse_event))
        }
        _ => None,
    });

    SseKeepAlive::new(stream).stream(res);
    Ok(())
}

/// POST /v1/models/download/cancel - Cancel in-progress download
#[handler]
pub async fn cancel_download(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    let request: CancelDownloadRequest = req.parse_json().await.map_err(|e| {
        tracing::error!("Failed to parse cancel request: {}", e);
        StatusError::bad_request()
    })?;

    match crate::download::cancel_download(&state.download_cancel_flags, &request.model_id) {
        Ok(()) => {
            res.render(Json(serde_json::json!({
                "status": "success",
                "message": format!("Download cancellation requested for '{}'", request.model_id),
            })));
        }
        Err(e) => {
            render_error(
                res,
                salvo::http::StatusCode::NOT_FOUND,
                &e,
                "not_found_error",
            );
        }
    }

    Ok(())
}

/// POST /v1/models/remove - Delete model files from disk
#[handler]
pub async fn remove_model(req: &mut Request, res: &mut Response) {
    let request: RemoveModelRequest = match req.parse_json().await {
        Ok(r) => r,
        Err(e) => {
            render_error(
                res,
                salvo::http::StatusCode::BAD_REQUEST,
                &format!("Invalid request: {}", e),
                "invalid_request_error",
            );
            return;
        }
    };

    // Run removal in a blocking thread (filesystem I/O)
    let model_id = request.model_id.clone();
    let result = tokio::task::spawn_blocking(move || {
        crate::download::remove_model(&model_id)
    })
    .await;

    match result {
        Ok(Ok(msg)) => {
            res.render(Json(serde_json::json!({
                "status": "success",
                "message": msg,
            })));
        }
        Ok(Err(e)) => {
            render_error(
                res,
                salvo::http::StatusCode::INTERNAL_SERVER_ERROR,
                &e,
                "remove_error",
            );
        }
        Err(e) => {
            render_error(
                res,
                salvo::http::StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Task failed: {}", e),
                "internal_error",
            );
        }
    }
}

/// POST /v1/models/scan - Rescan hub caches and filesystem
#[handler]
pub async fn scan_models(res: &mut Response) {
    let result = tokio::task::spawn_blocking(|| {
        crate::model_config::scan_hub_caches()
    })
    .await;

    match result {
        Ok(added) => {
            res.render(Json(serde_json::json!({
                "status": "success",
                "models_added": added,
                "message": format!("Scan complete, {} new model(s) registered", added),
            })));
        }
        Err(e) => {
            render_error(
                res,
                salvo::http::StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Scan failed: {}", e),
                "internal_error",
            );
        }
    }
}
