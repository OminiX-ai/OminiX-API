use serde::{Deserialize, Serialize};

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
