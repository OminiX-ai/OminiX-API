use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct VideoGenerationRequest {
    /// Text prompt for video generation.
    pub prompt: String,
    /// Model ID or converted MLX model directory.
    #[serde(default)]
    pub model: Option<String>,
    /// Optional reference image (base64 PNG/JPEG) for I2V/TI2V models.
    #[serde(default)]
    pub image: Option<String>,
    /// Optional negative prompt.
    #[serde(default)]
    pub negative_prompt: Option<String>,
    /// Number of videos to generate. The Python MLX MVP supports one video.
    #[serde(default = "default_n")]
    pub n: usize,
    /// Video size (e.g. "1280x704", "704x1280").
    #[serde(default = "default_size")]
    pub size: String,
    /// Response format. The MVP returns b64_json.
    #[serde(default = "default_response_format")]
    #[allow(dead_code)]
    pub response_format: String,
    /// Number of frames. Wan requires 4n+1.
    #[serde(default = "default_num_frames")]
    pub num_frames: usize,
    /// Diffusion steps.
    #[serde(default)]
    pub steps: Option<usize>,
    /// Guidance scale, either a single value or "low,high" for dual-model variants.
    #[serde(default)]
    pub guide_scale: Option<String>,
    /// Random seed.
    #[serde(default)]
    pub seed: Option<i64>,
    /// Scheduler: euler, dpm++, or unipc.
    #[serde(default)]
    pub scheduler: Option<String>,
    /// VAE tiling mode.
    #[serde(default)]
    pub tiling: Option<String>,
}

fn default_n() -> usize {
    1
}

fn default_size() -> String {
    "1280x704".to_string()
}

fn default_response_format() -> String {
    "b64_json".to_string()
}

fn default_num_frames() -> usize {
    81
}

#[derive(Debug, Serialize)]
pub struct VideoGenerationResponse {
    pub created: i64,
    pub data: Vec<VideoData>,
}

#[derive(Debug, Serialize)]
pub struct VideoData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}
