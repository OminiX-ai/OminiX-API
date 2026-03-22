use tokio::sync::{mpsc, oneshot};

use crate::types::*;

/// A chunk of streaming audio data.
#[derive(Debug)]
pub enum AudioChunk {
    /// Raw PCM i16 samples (little-endian, mono, model sample rate)
    Pcm(Vec<u8>),
    /// Generation finished successfully
    Done { total_samples: usize, duration_secs: f32 },
    /// Error during generation
    Error(String),
}

/// Request sent to the inference thread
pub enum InferenceRequest {
    Chat {
        request: ChatCompletionRequest,
        response_tx: oneshot::Sender<eyre::Result<ChatCompletionResponse>>,
    },
    Transcribe {
        request: TranscriptionRequest,
        /// Expected ASR backend: "qwen3-asr", "paraformer", or None for any.
        /// Set by model-specific endpoints (/v1/audio/asr/qwen3 etc.).
        expected_backend: Option<String>,
        response_tx: oneshot::Sender<eyre::Result<TranscriptionResponse>>,
    },
    Speech {
        request: SpeechRequest,
        response_tx: oneshot::Sender<eyre::Result<Vec<u8>>>,
    },
    /// Streaming TTS — sends audio chunks as they are generated
    SpeechStream {
        request: SpeechRequest,
        chunk_tx: mpsc::Sender<AudioChunk>,
    },
    /// Voice cloning TTS (dedicated endpoint, always uses Base model)
    SpeechClone {
        request: SpeechCloneRequest,
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
    /// Load/switch Qwen3-TTS model dynamically
    LoadQwen3TtsModel {
        model_dir: String,
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
    pub qwen3_tts: Option<String>,
    pub image: Option<String>,
    pub vlm: Option<String>,
}
