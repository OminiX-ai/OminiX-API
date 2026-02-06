use tokio::sync::{broadcast, mpsc};

use crate::download;
use crate::inference::InferenceRequest;
use crate::training;
use crate::types::{DownloadProgressEvent, TrainingProgressEvent};

/// Application state shared across HTTP handlers
#[derive(Clone)]
pub struct AppState {
    /// Channel to send inference requests
    pub inference_tx: mpsc::Sender<InferenceRequest>,
    /// Channel to send training requests
    pub training_tx: mpsc::Sender<training::TrainingRequest>,
    /// Broadcast channel for training progress events (Sender is Clone)
    pub progress_tx: broadcast::Sender<TrainingProgressEvent>,
    /// Shared cancel flag for the active training task
    pub cancel_flag: training::CancelFlag,
    /// Channel to send download requests
    pub download_tx: mpsc::Sender<download::DownloadRequest>,
    /// Broadcast channel for download progress events
    pub download_progress_tx: broadcast::Sender<DownloadProgressEvent>,
    /// Shared cancel flags for active downloads
    pub download_cancel_flags: download::DownloadCancelFlags,
}
