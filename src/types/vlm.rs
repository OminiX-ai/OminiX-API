use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct VlmCompletionRequest {
    /// Model ID (e.g. "moxin-vlm" or path)
    pub model: String,
    /// Base64-encoded image (PNG or JPEG)
    pub image: String,
    /// Text prompt for the VLM
    pub prompt: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct VlmCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub content: String,
    pub usage: VlmUsage,
}

#[derive(Debug, Serialize)]
pub struct VlmUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}
