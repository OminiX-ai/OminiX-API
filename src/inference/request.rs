use tokio::sync::oneshot;

use crate::types::*;

/// Request sent to the inference thread
pub enum InferenceRequest {
    Chat {
        request: ChatCompletionRequest,
        response_tx: oneshot::Sender<eyre::Result<ChatCompletionResponse>>,
    },
    Transcribe {
        request: TranscriptionRequest,
        response_tx: oneshot::Sender<eyre::Result<TranscriptionResponse>>,
    },
    Speech {
        request: SpeechRequest,
        response_tx: oneshot::Sender<eyre::Result<Vec<u8>>>,
    },
    Image {
        request: ImageGenerationRequest,
        response_tx: oneshot::Sender<eyre::Result<ImageGenerationResponse>>,
    },
    VlmCompletion {
        request: VlmCompletionRequest,
        response_tx: oneshot::Sender<eyre::Result<VlmCompletionResponse>>,
    },
    /// Load/switch LLM model dynamically
    LoadLlmModel {
        model_id: String,
        response_tx: oneshot::Sender<eyre::Result<String>>,
    },
    /// Load/switch ASR model dynamically
    LoadAsrModel {
        model_dir: String,
        response_tx: oneshot::Sender<eyre::Result<String>>,
    },
    /// Load/switch TTS model dynamically
    LoadTtsModel {
        ref_audio: String,
        response_tx: oneshot::Sender<eyre::Result<String>>,
    },
    /// Load/switch image generation model dynamically
    LoadImageModel {
        model_id: String,
        response_tx: oneshot::Sender<eyre::Result<String>>,
    },
    /// Load/switch VLM model dynamically
    LoadVlmModel {
        model_id: String,
        response_tx: oneshot::Sender<eyre::Result<String>>,
    },
    /// Unload a model to free memory
    UnloadModel {
        model_type: String,
        response_tx: oneshot::Sender<eyre::Result<String>>,
    },
    /// Get current model status
    GetModelStatus {
        response_tx: oneshot::Sender<ModelStatus>,
    },
    /// Reload voices.json after training completes
    ReloadVoices {
        response_tx: oneshot::Sender<eyre::Result<()>>,
    },
}

/// Current status of all models
#[derive(Clone, serde::Serialize)]
pub struct ModelStatus {
    pub llm: Option<String>,
    pub asr: Option<String>,
    pub tts: Option<String>,
    pub image: Option<String>,
    pub vlm: Option<String>,
}
