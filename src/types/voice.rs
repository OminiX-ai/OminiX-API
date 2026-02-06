use serde::Serialize;

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
