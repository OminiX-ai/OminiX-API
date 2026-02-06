use serde::{Deserialize, Serialize};

/// POST /v1/models/download request
#[derive(Debug, Deserialize)]
pub struct DownloadModelRequest {
    pub model_id: String,
}

/// POST /v1/models/download response
#[derive(Debug, Serialize)]
pub struct DownloadModelResponse {
    pub task_id: String,
    pub model_id: String,
    pub status: String,
    pub message: String,
}

/// POST /v1/models/download/cancel request
#[derive(Debug, Deserialize)]
pub struct CancelDownloadRequest {
    pub model_id: String,
}

/// POST /v1/models/remove request
#[derive(Debug, Deserialize)]
pub struct RemoveModelRequest {
    pub model_id: String,
}

/// Download state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DownloadState {
    Queued,
    Listing,
    Downloading,
    Converting,
    Ready,
    Cancelled,
    Error,
}

/// SSE download progress event (broadcast over channel)
#[derive(Debug, Clone, Serialize)]
pub struct DownloadProgressEvent {
    pub model_id: String,
    pub task_id: String,
    pub state: DownloadState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_file: Option<String>,
    pub current_file_index: usize,
    pub total_files: usize,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    /// Overall progress 0.0..1.0
    pub progress: f32,
    pub speed_bytes_per_sec: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eta_seconds: Option<u64>,
    pub message: String,
    pub is_complete: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}
