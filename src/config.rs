/// Configuration from environment
#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub llm_model: String,
    pub asr_model_dir: String,
    pub tts_ref_audio: String,
    pub image_model: String,
    pub vlm_model: String,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            port: std::env::var("PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8080),
            llm_model: std::env::var("LLM_MODEL")
                .unwrap_or_else(|_| "mlx-community/Mistral-7B-Instruct-v0.2-4bit".to_string()),
            asr_model_dir: std::env::var("ASR_MODEL_DIR")
                .unwrap_or_else(|_| "".to_string()),
            tts_ref_audio: std::env::var("TTS_REF_AUDIO")
                .unwrap_or_else(|_| "".to_string()),
            image_model: std::env::var("IMAGE_MODEL")
                .unwrap_or_else(|_| "".to_string()),
            vlm_model: std::env::var("VLM_MODEL")
                .unwrap_or_else(|_| "".to_string()),
        }
    }
}
