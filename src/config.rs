/// Configuration from environment variables and CLI arguments.
///
/// CLI arguments take precedence over environment variables.
/// Supported CLI args:
///   --port <N>
///   --asr-model <dir>
///   --tts-model <dir>       (maps to qwen3_tts_model_dir)
///   --tts-ref-audio <path>  (maps to tts_ref_audio)
///   --llm-model <id>
///   --image-model <id>
///   --vlm-model <id>
///   --models-dir <dir>      (informational, used by model_config scan)
#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub llm_model: String,
    pub asr_model_dir: String,
    pub tts_ref_audio: String,
    pub image_model: String,
    pub vlm_model: String,
    pub qwen3_tts_model_dir: String,
    /// Path to app manifest (`ominix.toml`) for requirement validation.
    pub app_manifest: Option<String>,
}

impl Config {
    pub fn from_env() -> Self {
        // Start with env var defaults
        let mut config = Self {
            port: std::env::var("PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8080),
            llm_model: std::env::var("LLM_MODEL")
                .unwrap_or_default(),
            asr_model_dir: std::env::var("ASR_MODEL_DIR")
                .unwrap_or_default(),
            tts_ref_audio: std::env::var("TTS_REF_AUDIO")
                .unwrap_or_default(),
            image_model: std::env::var("IMAGE_MODEL")
                .unwrap_or_default(),
            vlm_model: std::env::var("VLM_MODEL")
                .unwrap_or_default(),
            qwen3_tts_model_dir: std::env::var("QWEN3_TTS_MODEL_DIR")
                .unwrap_or_default(),
            app_manifest: std::env::var("OMINIX_APP_MANIFEST").ok(),
        };

        // Override with CLI arguments
        let args: Vec<String> = std::env::args().collect();
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--port" => {
                    if let Some(val) = args.get(i + 1) {
                        if let Ok(p) = val.parse::<u16>() {
                            config.port = p;
                        }
                        i += 1;
                    }
                }
                "--asr-model" => {
                    if let Some(val) = args.get(i + 1) {
                        config.asr_model_dir = val.clone();
                        i += 1;
                    }
                }
                "--tts-model" => {
                    if let Some(val) = args.get(i + 1) {
                        config.qwen3_tts_model_dir = val.clone();
                        i += 1;
                    }
                }
                "--tts-ref-audio" => {
                    if let Some(val) = args.get(i + 1) {
                        config.tts_ref_audio = val.clone();
                        i += 1;
                    }
                }
                "--llm-model" => {
                    if let Some(val) = args.get(i + 1) {
                        config.llm_model = val.clone();
                        i += 1;
                    }
                }
                "--image-model" => {
                    if let Some(val) = args.get(i + 1) {
                        config.image_model = val.clone();
                        i += 1;
                    }
                }
                "--vlm-model" => {
                    if let Some(val) = args.get(i + 1) {
                        config.vlm_model = val.clone();
                        i += 1;
                    }
                }
                "--models-dir" => {
                    // Consumed by model_config, not stored here
                    i += 1;
                }
                "--app-manifest" => {
                    if let Some(val) = args.get(i + 1) {
                        config.app_manifest = Some(val.clone());
                        i += 1;
                    }
                }
                _ => {}
            }
            i += 1;
        }

        config
    }

    /// Clear CLI model fields that are disallowed by server_config.
    /// This prevents startup-loading models the agent isn't supposed to serve.
    pub fn apply_server_config(&mut self, sc: &crate::server_config::ServerConfig) {
        let check = |field: &mut String, category: &str| {
            if field.is_empty() {
                return;
            }
            // Extract model name from path (last component) for matching
            let name = std::path::Path::new(field.as_str())
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| field.clone());
            if !sc.is_model_allowed(category, &name) {
                tracing::warn!(
                    "Skipping {} model '{}' — not in server_config allowlist",
                    category, name
                );
                field.clear();
            }
        };

        check(&mut self.llm_model, "llm");
        check(&mut self.asr_model_dir, "asr");
        check(&mut self.tts_ref_audio, "tts");
        check(&mut self.qwen3_tts_model_dir, "tts");
        check(&mut self.image_model, "image");
        check(&mut self.vlm_model, "vlm");
    }
}
