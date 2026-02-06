use serde::{Deserialize, Serialize};

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
