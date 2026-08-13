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
///   --asr-mode <mode>       off | interactive | conversational | offline

/// How aggressively ASR requests are batched before hitting the GPU.
///
/// Batching raises throughput because decode at batch 1 is memory-bandwidth
/// bound — the weights stream out of memory to produce a single token, so extra
/// sequences ride along nearly free. The cost is latency: every sequence in a
/// batch waits for the whole batch's step. These presets are the measured knees
/// of that trade-off on an M5 Max (see `docs/asr-batching.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsrMode {
    /// No batching. Lowest possible latency, lowest throughput.
    Off,
    /// Small batches — keeps latency inside a ~300 ms interactive budget.
    Interactive,
    /// Balanced. The default: roughly double the sessions of `off` for
    /// latency that is still a small fraction of the utterance being spoken.
    Conversational,
    /// Large batches for bulk transcription, where latency does not matter.
    Offline,
}

impl AsrMode {
    /// Maximum requests coalesced into one batch.
    pub fn max_batch(self) -> usize {
        match self {
            AsrMode::Off => 1,
            AsrMode::Interactive => 4,
            AsrMode::Conversational => 8,
            AsrMode::Offline => 32,
        }
    }

    /// How long to wait for more requests before running a partial batch.
    ///
    /// Kept short on purpose. At any load that justifies batching, enough
    /// requests are already queued that the window rarely elapses; it exists so
    /// a lone request at low load is not held for long.
    pub fn window(self) -> std::time::Duration {
        std::time::Duration::from_millis(match self {
            AsrMode::Off => 0,
            AsrMode::Interactive => 10,
            AsrMode::Conversational => 25,
            AsrMode::Offline => 150,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            AsrMode::Off => "off",
            AsrMode::Interactive => "interactive",
            AsrMode::Conversational => "conversational",
            AsrMode::Offline => "offline",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "1" => Some(AsrMode::Off),
            "interactive" | "low-latency" => Some(AsrMode::Interactive),
            "conversational" | "balanced" | "default" => Some(AsrMode::Conversational),
            "offline" | "batch" | "throughput" => Some(AsrMode::Offline),
            _ => None,
        }
    }
}

impl Default for AsrMode {
    fn default() -> Self {
        AsrMode::Conversational
    }
}

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
    /// ASR batching preset — `ASR_MODE` or `--asr-mode`.
    pub asr_mode: AsrMode,
    /// Overrides `asr_mode`'s batch size when set (`ASR_MAX_BATCH`).
    pub asr_max_batch: Option<usize>,
    /// Host silicon generation, detected at startup.
    pub chip: crate::chip::AppleChip,
    /// Explicit `ASR_USE_ANE` override. `None` means "follow the chip".
    pub asr_use_ane: Option<bool>,
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
            asr_mode: std::env::var("ASR_MODE")
                .ok()
                .and_then(|m| {
                    let parsed = AsrMode::parse(&m);
                    if parsed.is_none() {
                        tracing::warn!(
                            "Unknown ASR_MODE '{}' — expected off|interactive|conversational|offline; \
                             falling back to {}",
                            m,
                            AsrMode::default().as_str()
                        );
                    }
                    parsed
                })
                .unwrap_or_default(),
            asr_max_batch: std::env::var("ASR_MAX_BATCH")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&v| v >= 1),
            chip: crate::chip::AppleChip::detect(),
            asr_use_ane: std::env::var("ASR_USE_ANE")
                .ok()
                .and_then(|v| match v.trim().to_ascii_lowercase().as_str() {
                    "1" | "true" | "yes" | "on" => Some(true),
                    "0" | "false" | "no" | "off" => Some(false),
                    other => {
                        tracing::warn!(
                            "Unrecognised ASR_USE_ANE '{}' — expected a boolean; \
                             falling back to the per-chip default",
                            other
                        );
                        None
                    }
                }),
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
                "--asr-mode" => {
                    if let Some(val) = args.get(i + 1) {
                        match AsrMode::parse(val) {
                            Some(m) => config.asr_mode = m,
                            None => tracing::warn!(
                                "Unknown --asr-mode '{}' — expected \
                                 off|interactive|conversational|offline; keeping {}",
                                val,
                                config.asr_mode.as_str()
                            ),
                        }
                        i += 1;
                    }
                }
                "--asr-max-batch" => {
                    if let Some(val) = args.get(i + 1) {
                        match val.parse::<usize>() {
                            Ok(v) if v >= 1 => config.asr_max_batch = Some(v),
                            _ => tracing::warn!(
                                "Invalid --asr-max-batch '{}' — expected a positive integer",
                                val
                            ),
                        }
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

    /// Requests coalesced into one ASR batch: the mode's preset unless
    /// `ASR_MAX_BATCH` / `--asr-max-batch` overrides it.
    pub fn asr_max_batch(&self) -> usize {
        self.asr_max_batch.unwrap_or_else(|| self.asr_mode.max_batch()).max(1)
    }

    /// Whether ASR may place encoder work on the Neural Engine.
    ///
    /// Defaults to the host chip's answer — M5 and later only. Before M5 the ANE
    /// is not a worthwhile second unit for this workload, so it stays off no
    /// matter what the batching mode is. `ASR_USE_ANE` overrides in either
    /// direction, for benchmarking a generation the default excludes.
    ///
    /// Note: this expresses policy. The encoder offload path itself is not
    /// implemented yet, so today this only decides what the server reports and
    /// what a future offload scheduler will be permitted to do.
    pub fn asr_use_ane(&self) -> bool {
        self.asr_use_ane.unwrap_or_else(|| self.chip.ane_useful_for_asr())
    }

    /// One line describing the detected host and what it enables.
    pub fn describe_host(&self) -> String {
        let ane = if self.asr_use_ane() { "enabled" } else { "disabled" };
        let overridden = if self.asr_use_ane.is_some() { " (overridden)" } else { "" };
        format!(
            "chip={} gpu_neural_accelerators={} asr_ane={}{}",
            self.chip.as_str(),
            self.chip.has_gpu_neural_accelerators(),
            ane,
            overridden
        )
    }

    /// How long the inference thread waits for a batch to fill.
    pub fn asr_batch_window(&self) -> std::time::Duration {
        if self.asr_max_batch() <= 1 {
            std::time::Duration::ZERO
        } else {
            self.asr_mode.window()
        }
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
