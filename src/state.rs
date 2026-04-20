use std::sync::Arc;

use tokio::sync::{broadcast, mpsc};

use crate::download;
use crate::engines::ascend::AscendConfig;
use crate::engines::tts_trait::TextToSpeech;
use crate::inference::InferenceRequest;
use crate::server_config::ServerConfig;
use crate::training;
use crate::types::{DownloadProgressEvent, TrainingProgressEvent};

/// Application state shared across HTTP handlers
#[derive(Clone)]
pub struct AppState {
    /// Channel to send ALL inference requests (LLM, ASR, TTS, image, VLM)
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
    /// Server config (model gatekeeper)
    pub server_config: Arc<ServerConfig>,
    /// Optional Ascend NPU backend configuration
    pub ascend_config: Option<Arc<AscendConfig>>,
    /// Shared TTS backend for Ascend endpoints (selected at startup via
    /// `ASCEND_TTS_TRANSPORT`: `ffi`|`subprocess`, default `subprocess`).
    /// `None` when `ascend_config` is `None`.
    ///
    /// See `ASCEND_API_BRIDGE_CONTRACT.md` §5 B3.3.
    pub ascend_tts_backend: Option<Arc<dyn TextToSpeech>>,
}
