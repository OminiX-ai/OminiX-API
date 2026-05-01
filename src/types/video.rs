use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct VideoGenerationRequest {
    pub prompt: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_size")]
    pub size: String,
    #[serde(default = "default_num_frames")]
    pub num_frames: i32,
    #[serde(default = "default_steps")]
    pub steps: i32,
    #[serde(default = "default_response_format")]
    #[allow(dead_code)]
    pub response_format: String,
}

fn default_size() -> String {
    "480x320".to_string()
}

fn default_num_frames() -> i32 {
    49
}

fn default_steps() -> i32 {
    20
}

fn default_response_format() -> String {
    "b64_json".to_string()
}

#[derive(Debug, Serialize)]
pub struct VideoGenerationResponse {
    pub created: i64,
    pub data: Vec<VideoFrameData>,
}

#[derive(Debug, Serialize)]
pub struct VideoFrameData {
    pub frame_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
}
