use std::time::Duration;

use salvo::prelude::*;
use tokio::sync::{mpsc, oneshot};
use tokio::time::timeout;

use crate::inference::InferenceRequest;
use crate::state::AppState;

pub(crate) fn get_state(depot: &mut Depot) -> Result<&AppState, StatusError> {
    depot
        .obtain::<AppState>()
        .map_err(|_| StatusError::internal_server_error())
}

/// Send an inference request and wait for the response with a timeout.
///
/// `make_request` receives a oneshot sender and returns the InferenceRequest variant.
pub(crate) async fn send_and_wait<T>(
    tx: &mpsc::Sender<InferenceRequest>,
    make_request: impl FnOnce(oneshot::Sender<eyre::Result<T>>) -> InferenceRequest,
    timeout_duration: Duration,
) -> Result<T, StatusError> {
    let (response_tx, response_rx) = oneshot::channel();
    tx.send(make_request(response_tx))
        .await
        .map_err(|_| StatusError::internal_server_error())?;

    timeout(timeout_duration, response_rx)
        .await
        .map_err(|_| StatusError::gateway_timeout())?
        .map_err(|_| StatusError::internal_server_error())?
        .map_err(|e| {
            tracing::error!("Inference error: {}", e);
            StatusError::internal_server_error()
        })
}
