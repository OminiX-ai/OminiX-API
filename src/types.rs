//! OpenAI-compatible request/response types
//!
//! Some fields are deserialized for API compatibility but not yet used by
//! the inference backends (e.g., `top_p`, `response_format`, `speed`).

use serde::{Deserialize, Serialize};

// ============================================================================
// Chat Completions
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    #[allow(dead_code)]
    pub top_p: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: ChatUsage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ============================================================================
// Audio Transcriptions (ASR)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct TranscriptionRequest {
    /// Base64-encoded audio data or URL
    pub file: String,
    /// Model to use (ignored, we use paraformer)
    #[serde(default)]
    #[allow(dead_code)]
    pub model: Option<String>,
    /// Language hint (optional)
    #[serde(default)]
    pub language: Option<String>,
    /// Response format: "json", "text", "srt", "verbose_json", "vtt"
    #[serde(default)]
    #[allow(dead_code)]
    pub response_format: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TranscriptionResponse {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f32>,
}

// ============================================================================
// Audio Speech (TTS)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    /// Model to use (ignored, we use GPT-SoVITS)
    #[serde(default)]
    #[allow(dead_code)]
    pub model: Option<String>,
    /// Text to synthesize
    pub input: String,
    /// Voice/speaker ID (optional, for multi-speaker models)
    #[serde(default)]
    pub voice: Option<String>,
    /// Response format: "mp3", "opus", "aac", "flac", "wav", "pcm"
    #[serde(default = "default_audio_format")]
    #[allow(dead_code)]
    pub response_format: String,
    /// Speaking speed (0.25 to 4.0)
    #[serde(default = "default_speed")]
    #[allow(dead_code)]
    pub speed: f32,
}

fn default_audio_format() -> String {
    "wav".to_string()
}

fn default_speed() -> f32 {
    1.0
}

// ============================================================================
// Image Generation
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ImageGenerationRequest {
    /// Text prompt for image generation
    pub prompt: String,
    /// Model to use (ignored, we use FLUX)
    #[serde(default)]
    pub model: Option<String>,
    /// Number of images to generate
    #[serde(default = "default_n")]
    pub n: usize,
    /// Image size (e.g., "512x512", "1024x1024")
    #[serde(default = "default_size")]
    pub size: String,
    /// Response format: "url" or "b64_json"
    #[serde(default = "default_response_format")]
    #[allow(dead_code)]
    pub response_format: String,
    /// Quality: "standard" or "hd"
    #[serde(default)]
    #[allow(dead_code)]
    pub quality: Option<String>,
    /// Reference image (base64 encoded PNG/JPEG) for img2img
    #[serde(default)]
    pub image: Option<String>,
    /// Strength for img2img (0.0-1.0, higher = more variation from reference)
    #[serde(default = "default_strength")]
    pub strength: f32,
}

fn default_strength() -> f32 {
    0.75
}

fn default_n() -> usize {
    1
}

fn default_size() -> String {
    "512x512".to_string()
}

fn default_response_format() -> String {
    "b64_json".to_string()
}

#[derive(Debug, Serialize)]
pub struct ImageGenerationResponse {
    pub created: i64,
    pub data: Vec<ImageData>,
}

#[derive(Debug, Serialize)]
pub struct ImageData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}

// ============================================================================
// Voice Cloning Training
// ============================================================================

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

/// GET /v1/voices — List registered voices
#[derive(Debug, Serialize)]
pub struct VoiceListResponse {
    pub voices: Vec<VoiceInfo>,
}

#[derive(Debug, Serialize)]
pub struct VoiceInfo {
    pub name: String,
    pub aliases: Vec<String>,
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

// ============================================================================
// Standardized API Error Response
// ============================================================================

/// OpenAI-compatible error response envelope
#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: ApiErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorDetail {
    pub message: String,
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}
