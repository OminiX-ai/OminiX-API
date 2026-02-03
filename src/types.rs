//! OpenAI-compatible request/response types
//!
//! Some fields are deserialized for API compatibility but not yet used by
//! the inference backends (e.g., `top_p`, `response_format`, `speed`).

#![allow(dead_code)]

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
    pub model: Option<String>,
    /// Language hint (optional)
    #[serde(default)]
    pub language: Option<String>,
    /// Response format: "json", "text", "srt", "verbose_json", "vtt"
    #[serde(default)]
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
    pub model: Option<String>,
    /// Text to synthesize
    pub input: String,
    /// Voice/speaker ID (optional, for multi-speaker models)
    #[serde(default)]
    pub voice: Option<String>,
    /// Response format: "mp3", "opus", "aac", "flac", "wav", "pcm"
    #[serde(default = "default_audio_format")]
    pub response_format: String,
    /// Speaking speed (0.25 to 4.0)
    #[serde(default = "default_speed")]
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
    pub response_format: String,
    /// Quality: "standard" or "hd"
    #[serde(default)]
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
