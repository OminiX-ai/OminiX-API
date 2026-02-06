use serde::{Deserialize, Serialize};

/// Quality preset for voice cloning training
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TrainingQuality {
    Fast,
    Standard,
    High,
}

impl Default for TrainingQuality {
    fn default() -> Self {
        Self::Standard
    }
}

/// POST /v1/voices/train — Start voice cloning training
#[derive(Debug, Deserialize)]
pub struct VoiceTrainRequest {
    /// Voice name (used for registration in voices.json)
    pub voice_name: String,
    /// Base64-encoded reference audio (WAV, MP3, FLAC)
    pub audio: String,
    /// Transcript of the reference audio
    pub transcript: String,
    /// Quality preset (fast/standard/high)
    #[serde(default)]
    pub quality: TrainingQuality,
    /// Language hint
    #[serde(default = "default_training_language")]
    pub language: String,
    /// Enable audio denoising
    #[serde(default)]
    pub denoise: bool,
}

fn default_training_language() -> String {
    "zh".to_string()
}

/// Response to POST /v1/voices/train
#[derive(Debug, Serialize)]
pub struct VoiceTrainResponse {
    pub task_id: String,
    pub status: String,
    pub message: String,
}

/// Training pipeline stage
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingStage {
    Queued,
    AudioSlicing,
    Denoising,
    FeatureExtraction,
    VitsTraining,
    RegisteringVoice,
    Complete,
    Failed,
}

/// SSE progress event sent during training
#[derive(Debug, Clone, Serialize)]
pub struct TrainingProgressEvent {
    pub task_id: String,
    pub stage: TrainingStage,
    /// Overall progress (0.0 to 1.0)
    pub progress: f32,
    /// Current stage progress (0.0 to 1.0)
    pub stage_progress: f32,
    /// Human-readable message
    pub message: String,
    /// Loss values during training stages
    #[serde(skip_serializing_if = "Option::is_none")]
    pub losses: Option<TrainingLossInfo>,
    /// Is this the final event?
    pub is_complete: bool,
    /// Error message if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainingLossInfo {
    pub epoch: usize,
    pub step: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loss_total: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loss_mel: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loss_kl: Option<f32>,
}

/// GET /v1/voices/train/{task_id} — Training task status
#[derive(Debug, Clone, Serialize)]
pub struct TrainingTaskStatus {
    pub task_id: String,
    pub voice_name: String,
    pub status: TrainingStage,
    pub progress: f32,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}
