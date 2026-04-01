//! Ascend NPU inference backend
//!
//! Calls OminiX-Ascend C++ binaries for inference on Huawei Ascend 910B NPUs.
//! Same integration pattern as MLX — the binaries run on the same machine.
//! Models are in GGUF format.
//!
//! Supported models:
//! - **LLM**: Chat completions via llama-server HTTP proxy (100+ architectures)
//! - **VLM**: Vision-language via llama-server with mmproj
//! - **Qwen3-ASR**: Speech recognition (qwen_asr binary)
//! - **Qwen3-TTS**: Text-to-speech with voice cloning (qwen_tts binary)
//! - **OuteTTS**: Alternative TTS via llama-tts binary
//! - **Image**: Generation/editing via ominix-diffusion-cli or diffusion-server

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;

use eyre::{Context, Result};

use crate::types::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ChatUsage,
    ImageData, ImageGenerationRequest, ImageGenerationResponse,
    SpeechRequest, TranscriptionRequest, TranscriptionResponse,
    VlmCompletionRequest, VlmCompletionResponse, VlmUsage,
};

// ============================================================================
// Configuration
// ============================================================================

/// Ascend backend configuration — paths to binaries and model weights.
#[derive(Debug, Clone)]
pub struct AscendConfig {
    /// Path to the OminiX-Ascend build/bin directory
    pub bin_dir: PathBuf,
    // --- LLM ---
    /// Path to LLM GGUF model file (for llama-server)
    pub llm_model: Option<PathBuf>,
    /// Port for llama-server LLM (default: 8081)
    pub llm_server_port: u16,
    // --- VLM ---
    /// Path to VLM GGUF model file
    pub vlm_model: Option<PathBuf>,
    /// Path to multimodal projector GGUF (for vision)
    pub vlm_mmproj: Option<PathBuf>,
    /// Port for llama-server VLM (default: 8082)
    pub vlm_server_port: u16,
    // --- ASR ---
    /// Path to ASR GGUF model directory
    pub asr_model_dir: Option<PathBuf>,
    // --- TTS ---
    /// Path to TTS GGUF model directory
    pub tts_model_dir: Option<PathBuf>,
    /// Path to TTS CustomVoice GGUF model directory (preset voices)
    pub tts_cv_model_dir: Option<PathBuf>,
    // --- OuteTTS ---
    /// Path to OuteTTS LLM model GGUF
    pub outetts_model: Option<PathBuf>,
    /// Path to OuteTTS vocoder (WavTokenizer) GGUF
    pub outetts_vocoder: Option<PathBuf>,
    // --- Diffusion ---
    /// Path to diffusion model files (GGUF + VAE + LLM)
    pub diffusion_model: Option<PathBuf>,
    pub diffusion_vae: Option<PathBuf>,
    pub diffusion_llm: Option<PathBuf>,
    /// Port for ominix-diffusion-server (default: 8083)
    pub diffusion_server_port: u16,
    // --- General ---
    /// Number of GPU layers to offload (default: 29 for full offload)
    pub gpu_layers: u32,
    /// Number of CPU threads (default: 8)
    pub threads: u32,
    /// Reference audio files directory for built-in TTS voices
    pub voices_dir: Option<PathBuf>,
    /// Environment setup: LD_LIBRARY_PATH additions for CANN
    pub cann_env: AscendCannEnv,
}

/// CANN environment variables needed to run Ascend binaries.
#[derive(Debug, Clone)]
pub struct AscendCannEnv {
    pub ascend_toolkit_home: String,
    pub driver_lib_dirs: Vec<String>,
}

impl Default for AscendCannEnv {
    fn default() -> Self {
        Self {
            ascend_toolkit_home: "/usr/local/Ascend/ascend-toolkit/latest".to_string(),
            driver_lib_dirs: vec![
                "/usr/local/Ascend/driver/lib64".to_string(),
                "/usr/local/Ascend/driver/lib64/common".to_string(),
                "/usr/local/Ascend/driver/lib64/driver".to_string(),
            ],
        }
    }
}

impl AscendConfig {
    /// Build from environment variables.
    pub fn from_env() -> Option<Self> {
        let bin_dir = std::env::var("ASCEND_BIN_DIR").ok()?;
        let env_path = |key: &str| std::env::var(key).ok().map(PathBuf::from);
        let env_port = |key: &str, default: u16| -> u16 {
            std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
        };

        Some(Self {
            bin_dir: PathBuf::from(&bin_dir),
            llm_model: env_path("ASCEND_LLM_MODEL"),
            llm_server_port: env_port("ASCEND_LLM_PORT", 8081),
            vlm_model: env_path("ASCEND_VLM_MODEL"),
            vlm_mmproj: env_path("ASCEND_VLM_MMPROJ"),
            vlm_server_port: env_port("ASCEND_VLM_PORT", 8082),
            asr_model_dir: env_path("ASCEND_ASR_MODEL_DIR"),
            tts_model_dir: env_path("ASCEND_TTS_MODEL_DIR"),
            tts_cv_model_dir: env_path("ASCEND_TTS_CV_MODEL_DIR"),
            outetts_model: env_path("ASCEND_OUTETTS_MODEL"),
            outetts_vocoder: env_path("ASCEND_OUTETTS_VOCODER"),
            diffusion_model: env_path("ASCEND_DIFFUSION_MODEL"),
            diffusion_vae: env_path("ASCEND_DIFFUSION_VAE"),
            diffusion_llm: env_path("ASCEND_DIFFUSION_LLM"),
            diffusion_server_port: env_port("ASCEND_DIFFUSION_PORT", 8083),
            gpu_layers: std::env::var("ASCEND_GPU_LAYERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(29),
            threads: std::env::var("ASCEND_THREADS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8),
            voices_dir: std::env::var("ASCEND_VOICES_DIR").ok().map(PathBuf::from),
            cann_env: AscendCannEnv::default(),
        })
    }

    pub fn has_llm(&self) -> bool {
        self.llm_model.as_ref().is_some_and(|p| p.exists())
            && self.bin_dir.join("llama-server").exists()
    }

    pub fn has_vlm(&self) -> bool {
        self.vlm_model.as_ref().is_some_and(|p| p.exists())
            && self.vlm_mmproj.as_ref().is_some_and(|p| p.exists())
            && self.bin_dir.join("llama-server").exists()
    }

    pub fn has_asr(&self) -> bool {
        self.asr_model_dir.is_some() && self.bin_dir.join("qwen_asr").exists()
    }

    pub fn has_tts(&self) -> bool {
        self.tts_model_dir.is_some() && self.bin_dir.join("qwen_tts").exists()
    }

    pub fn has_outetts(&self) -> bool {
        self.outetts_model.as_ref().is_some_and(|p| p.exists())
            && self.outetts_vocoder.as_ref().is_some_and(|p| p.exists())
            && self.bin_dir.join("llama-tts").exists()
    }

    pub fn has_diffusion(&self) -> bool {
        self.diffusion_model.is_some()
            && (self.bin_dir.join("ominix-diffusion-cli").exists()
                || self.bin_dir.join("ominix-diffusion-server").exists())
    }

    /// Build LD_LIBRARY_PATH for running Ascend binaries.
    fn ld_library_path(&self) -> String {
        let mut paths = self.cann_env.driver_lib_dirs.clone();
        paths.push(format!("{}/lib64", self.cann_env.ascend_toolkit_home));
        paths.push(format!(
            "{}/lib64/plugin/opskernel",
            self.cann_env.ascend_toolkit_home
        ));
        // Include the bin dir itself for shared libraries
        paths.push(self.bin_dir.to_string_lossy().to_string());
        // Append existing LD_LIBRARY_PATH
        if let Ok(existing) = std::env::var("LD_LIBRARY_PATH") {
            paths.push(existing);
        }
        paths.join(":")
    }

    /// Create a Command with proper Ascend environment set up.
    fn command(&self, binary: &str) -> Command {
        let bin_path = self.bin_dir.join(binary);
        let mut cmd = Command::new(&bin_path);
        cmd.env("LD_LIBRARY_PATH", self.ld_library_path());
        cmd.env("ASCEND_TOOLKIT_HOME", &self.cann_env.ascend_toolkit_home);
        cmd
    }
}

// ============================================================================
// ASR Engine
// ============================================================================

/// Ascend-based ASR engine using qwen_asr binary.
pub struct AscendAsrEngine {
    config: AscendConfig,
}

impl AscendAsrEngine {
    pub fn new(config: AscendConfig) -> Result<Self> {
        let model_dir = config
            .asr_model_dir
            .as_ref()
            .ok_or_else(|| eyre::eyre!("ASCEND_ASR_MODEL_DIR not set"))?;

        // Verify required files exist
        let encoder = model_dir.join("qwen_asr_audio_encoder.gguf");
        let decoder = find_asr_decoder(model_dir)?;
        if !encoder.exists() {
            eyre::bail!("ASR encoder not found: {}", encoder.display());
        }

        // Verify tokenizer files
        let vocab = model_dir.join("vocab.json");
        let merges = model_dir.join("merges.txt");
        if !vocab.exists() || !merges.exists() {
            eyre::bail!("ASR tokenizer files (vocab.json, merges.txt) not found in {}", model_dir.display());
        }

        tracing::info!(
            "Ascend ASR ready: encoder={}, decoder={}",
            encoder.display(),
            decoder.display()
        );

        Ok(Self { config })
    }

    pub fn backend_name(&self) -> &'static str {
        "ascend-qwen3-asr"
    }

    /// Transcribe audio to text by running the qwen_asr binary.
    pub fn transcribe(&self, request: &TranscriptionRequest) -> Result<TranscriptionResponse> {
        let model_dir = self.config.asr_model_dir.as_ref().unwrap();
        let encoder = model_dir.join("qwen_asr_audio_encoder.gguf");
        let decoder = find_asr_decoder(model_dir)?;

        // Decode audio to WAV temp file
        let audio_bytes = decode_request_audio(&request.file)?;
        let tmp_wav = write_wav_tempfile(&audio_bytes)?;

        let mut cmd = self.config.command("qwen_asr");
        cmd.args([
            "--audio",
            tmp_wav.path().to_str().unwrap(),
            "--model_dir",
            model_dir.to_str().unwrap(),
            "--encoder",
            encoder.to_str().unwrap(),
            "--decoder",
            decoder.to_str().unwrap(),
            "--gpu_layers",
            &self.config.gpu_layers.to_string(),
            "--threads",
            &self.config.threads.to_string(),
        ]);
        // Add tokenizer_config.json if available
        let tokenizer_config = model_dir.join("tokenizer_config.json");
        if tokenizer_config.exists() {
            // tokenizer_config is auto-detected from model_dir
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        tracing::info!("Running Ascend ASR: {:?}", cmd);
        let output = cmd
            .output()
            .context("Failed to run qwen_asr binary")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eyre::bail!("qwen_asr failed: {}", stderr.lines().last().unwrap_or("unknown"));
        }

        // Parse output — the result text follows "=== Result ===" line
        let stdout = String::from_utf8_lossy(&output.stdout);
        let text = parse_asr_output(&stdout);

        // Extract timing from stderr for duration info
        let stderr = String::from_utf8_lossy(&output.stderr);
        let duration = parse_asr_duration(&stderr);

        Ok(TranscriptionResponse {
            text,
            language: request.language.clone(),
            duration,
        })
    }
}

// ============================================================================
// TTS Engine
// ============================================================================

/// Ascend-based TTS engine using qwen_tts binary.
pub struct AscendTtsEngine {
    config: AscendConfig,
}

impl AscendTtsEngine {
    pub fn new(config: AscendConfig) -> Result<Self> {
        let model_dir = config
            .tts_model_dir
            .as_ref()
            .ok_or_else(|| eyre::eyre!("ASCEND_TTS_MODEL_DIR not set"))?;

        // Verify core model files
        let talker = model_dir.join("qwen_tts_talker.gguf");
        if !talker.exists() {
            eyre::bail!("TTS talker model not found: {}", talker.display());
        }

        tracing::info!("Ascend TTS ready: model_dir={}", model_dir.display());
        Ok(Self { config })
    }

    /// Synthesize speech with a built-in/preset voice.
    ///
    /// Uses a pre-cached reference audio from the voices directory.
    pub fn synthesize(&self, request: &SpeechRequest) -> Result<Vec<u8>> {
        let model_dir = self.config.tts_model_dir.as_ref().unwrap();

        // Determine voice reference audio
        let voice = request.voice.as_deref().unwrap_or("default");
        let (ref_audio, ref_text) = self.resolve_voice(voice)?;

        let language = request.language.as_deref().unwrap_or("English");
        self.run_tts(model_dir, &request.input, language, &ref_audio, &ref_text, language)
    }

    /// Synthesize speech with voice cloning from reference audio bytes.
    pub fn synthesize_clone(
        &self,
        text: &str,
        ref_audio_bytes: &[u8],
        language: &str,
        _speed: f32,
        _instruct: Option<&str>,
    ) -> Result<Vec<u8>> {
        let model_dir = self.config.tts_model_dir.as_ref().unwrap();

        // Write reference audio to temp file
        let tmp_ref = write_audio_tempfile(ref_audio_bytes, ".wav")?;

        // We don't know the ref text for clone, use empty string
        // (the C++ binary will encode the reference audio anyway)
        let ref_text = "reference audio";

        self.run_tts(
            model_dir,
            text,
            language,
            tmp_ref.path().to_str().unwrap(),
            ref_text,
            language,
        )
    }

    /// Resolve a voice name to (ref_audio_path, ref_text).
    fn resolve_voice(&self, voice: &str) -> Result<(String, String)> {
        // Check voices directory for matching reference audio
        if let Some(ref voices_dir) = self.config.voices_dir {
            // Try common naming patterns
            for ext in &["wav", "mp3"] {
                let audio_path = voices_dir.join(format!("{}_ref.{}", voice, ext));
                if audio_path.exists() {
                    let text_path = voices_dir.join(format!("{}_ref.txt", voice));
                    let ref_text = std::fs::read_to_string(&text_path).unwrap_or_default();
                    return Ok((audio_path.to_string_lossy().to_string(), ref_text));
                }
                let audio_path = voices_dir.join(format!("{}.{}", voice, ext));
                if audio_path.exists() {
                    let text_path = voices_dir.join(format!("{}.txt", voice));
                    let ref_text = std::fs::read_to_string(&text_path).unwrap_or_default();
                    return Ok((audio_path.to_string_lossy().to_string(), ref_text));
                }
            }
        }

        // Fall back to first available reference audio in the TTS model dir
        if let Some(ref model_dir) = self.config.tts_model_dir {
            if let Ok(entries) = std::fs::read_dir(model_dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.ends_with("_ref.wav") {
                        return Ok((entry.path().to_string_lossy().to_string(), String::new()));
                    }
                }
            }
        }

        eyre::bail!(
            "Voice '{}' not found. Place ref audio in ASCEND_VOICES_DIR as {}_ref.wav",
            voice,
            voice
        )
    }

    /// Run the qwen_tts binary and return WAV audio bytes.
    fn run_tts(
        &self,
        model_dir: &Path,
        text: &str,
        target_lang: &str,
        ref_audio: &str,
        ref_text: &str,
        ref_lang: &str,
    ) -> Result<Vec<u8>> {
        let tmp_output = tempfile::Builder::new()
            .suffix(".wav")
            .tempfile()
            .context("Failed to create temp output file")?;

        let talker_q8 = model_dir.join("qwen_tts_talker_llama_q8_0.gguf");
        let cp_llama = model_dir.join("qwen_tts_cp_llama.gguf");

        let mut cmd = self.config.command("qwen_tts");
        cmd.args([
            "-m",
            model_dir.to_str().unwrap(),
            "-t",
            text,
            "-r",
            ref_audio,
            "--ref_text",
            ref_text,
            "--target_lang",
            target_lang,
            "--ref_lang",
            ref_lang,
            "-o",
            tmp_output.path().to_str().unwrap(),
            "--n_gpu_layers",
            &self.config.gpu_layers.to_string(),
            "-n",
            &self.config.threads.to_string(),
        ]);

        // Use quantized talker if available
        if talker_q8.exists() {
            cmd.args(["--talker_model", talker_q8.to_str().unwrap()]);
        }
        // Use CP llama for NPU acceleration if available
        if cp_llama.exists() {
            cmd.args(["--cp_model", cp_llama.to_str().unwrap()]);
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        tracing::info!("Running Ascend TTS: text='{}', lang={}", text, target_lang);
        let output = cmd.output().context("Failed to run qwen_tts binary")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eyre::bail!("qwen_tts failed: {}", stderr.lines().last().unwrap_or("unknown"));
        }

        // Read the generated WAV file
        let wav_data = std::fs::read(tmp_output.path())
            .context("Failed to read TTS output WAV")?;

        if wav_data.is_empty() {
            eyre::bail!("qwen_tts produced empty output");
        }

        tracing::info!("Ascend TTS complete: {} bytes WAV", wav_data.len());
        Ok(wav_data)
    }
}

// ============================================================================
// LLM Engine (via llama-server HTTP proxy)
// ============================================================================

/// Manages a llama-server child process and proxies OpenAI-compatible requests.
///
/// The server is started on first use and kept running for subsequent requests.
/// Uses the same `/v1/chat/completions` endpoint that llama-server exposes.
pub struct AscendLlmServer {
    config: AscendConfig,
    /// The running llama-server child process (if started)
    child: Mutex<Option<std::process::Child>>,
    port: u16,
}

impl AscendLlmServer {
    pub fn new(config: AscendConfig) -> Result<Self> {
        let model = config
            .llm_model
            .as_ref()
            .ok_or_else(|| eyre::eyre!("ASCEND_LLM_MODEL not set"))?;
        if !model.exists() {
            eyre::bail!("LLM model not found: {}", model.display());
        }
        let port = config.llm_server_port;
        tracing::info!("Ascend LLM ready: model={}, port={}", model.display(), port);
        Ok(Self {
            config,
            child: Mutex::new(None),
            port,
        })
    }

    /// Ensure the llama-server process is running, start if needed.
    pub fn ensure_running(&self) -> Result<()> {
        let mut child_guard = self.child.lock().map_err(|e| eyre::eyre!("lock: {e}"))?;

        // Check if already running
        if let Some(ref mut child) = *child_guard {
            match child.try_wait() {
                Ok(None) => return Ok(()), // still running
                Ok(Some(status)) => {
                    tracing::warn!("llama-server exited with {status}, restarting...");
                }
                Err(e) => {
                    tracing::warn!("Failed to check llama-server status: {e}, restarting...");
                }
            }
        }

        let model = self.config.llm_model.as_ref().unwrap();
        let mut cmd = self.config.command("llama-server");
        cmd.args([
            "-m",
            model.to_str().unwrap(),
            "--host",
            "127.0.0.1",
            "--port",
            &self.port.to_string(),
            "-ngl",
            &self.config.gpu_layers.to_string(),
            "-t",
            &self.config.threads.to_string(),
            "-fa",
            // Flash attention
        ]);
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::piped());

        tracing::info!("Starting llama-server on port {}", self.port);
        let child_proc = cmd.spawn().context("Failed to start llama-server")?;
        *child_guard = Some(child_proc);

        // Wait for server to be ready (poll /health)
        drop(child_guard);
        self.wait_for_health(30)?;

        Ok(())
    }

    fn wait_for_health(&self, timeout_secs: u64) -> Result<()> {
        let url = format!("http://127.0.0.1:{}/health", self.port);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
        while std::time::Instant::now() < deadline {
            if let Ok(resp) = reqwest::blocking::get(&url) {
                if resp.status().is_success() {
                    tracing::info!("llama-server healthy on port {}", self.port);
                    return Ok(());
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        eyre::bail!("llama-server did not become healthy within {timeout_secs}s")
    }

    /// Proxy a chat completion request to the running llama-server.
    pub fn chat_completions(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        self.ensure_running()?;

        let url = format!("http://127.0.0.1:{}/v1/chat/completions", self.port);
        let client = reqwest::blocking::Client::new();
        let resp = client
            .post(&url)
            .json(request)
            .timeout(std::time::Duration::from_secs(300))
            .send()
            .context("Failed to send request to llama-server")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            eyre::bail!("llama-server returned {status}: {body}");
        }

        let response: ChatCompletionResponse = resp.json().context("Failed to parse llama-server response")?;
        Ok(response)
    }
}

impl Drop for AscendLlmServer {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.child.lock() {
            if let Some(ref mut child) = *guard {
                let _ = child.kill();
                let _ = child.wait();
                tracing::info!("llama-server (LLM, port {}) stopped", self.port);
            }
        }
    }
}

// ============================================================================
// VLM Engine (via llama-server with multimodal projector)
// ============================================================================

/// Manages a llama-server child process with vision capabilities.
pub struct AscendVlmServer {
    config: AscendConfig,
    child: Mutex<Option<std::process::Child>>,
    port: u16,
}

impl AscendVlmServer {
    pub fn new(config: AscendConfig) -> Result<Self> {
        let model = config
            .vlm_model
            .as_ref()
            .ok_or_else(|| eyre::eyre!("ASCEND_VLM_MODEL not set"))?;
        let mmproj = config
            .vlm_mmproj
            .as_ref()
            .ok_or_else(|| eyre::eyre!("ASCEND_VLM_MMPROJ not set"))?;
        if !model.exists() {
            eyre::bail!("VLM model not found: {}", model.display());
        }
        if !mmproj.exists() {
            eyre::bail!("VLM mmproj not found: {}", mmproj.display());
        }
        let port = config.vlm_server_port;
        tracing::info!("Ascend VLM ready: model={}, mmproj={}", model.display(), mmproj.display());
        Ok(Self {
            config,
            child: Mutex::new(None),
            port,
        })
    }

    pub fn ensure_running(&self) -> Result<()> {
        let mut child_guard = self.child.lock().map_err(|e| eyre::eyre!("lock: {e}"))?;

        if let Some(ref mut child) = *child_guard {
            match child.try_wait() {
                Ok(None) => return Ok(()),
                _ => {}
            }
        }

        let model = self.config.vlm_model.as_ref().unwrap();
        let mmproj = self.config.vlm_mmproj.as_ref().unwrap();
        let mut cmd = self.config.command("llama-server");
        cmd.args([
            "-m",
            model.to_str().unwrap(),
            "--mmproj",
            mmproj.to_str().unwrap(),
            "--host",
            "127.0.0.1",
            "--port",
            &self.port.to_string(),
            "-ngl",
            &self.config.gpu_layers.to_string(),
            "-t",
            &self.config.threads.to_string(),
        ]);
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::piped());

        tracing::info!("Starting llama-server (VLM) on port {}", self.port);
        let child_proc = cmd.spawn().context("Failed to start llama-server for VLM")?;
        *child_guard = Some(child_proc);
        drop(child_guard);

        // Wait for health
        let url = format!("http://127.0.0.1:{}/health", self.port);
        let deadline =
            std::time::Instant::now() + std::time::Duration::from_secs(60);
        while std::time::Instant::now() < deadline {
            if let Ok(resp) = reqwest::blocking::get(&url) {
                if resp.status().is_success() {
                    tracing::info!("llama-server (VLM) healthy on port {}", self.port);
                    return Ok(());
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        eyre::bail!("llama-server (VLM) did not become healthy within 60s")
    }

    /// Send a VLM completion request with an image.
    ///
    /// Converts our VlmCompletionRequest to a chat completion with image content,
    /// which llama-server's multimodal support handles.
    pub fn vlm_completion(
        &self,
        request: &VlmCompletionRequest,
    ) -> Result<VlmCompletionResponse> {
        self.ensure_running()?;

        // Build a chat message with image_url content (base64 data URL)
        let image_url = format!("data:image/png;base64,{}", request.image);
        let body = serde_json::json!({
            "model": request.model,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": { "url": image_url }
                    },
                    {
                        "type": "text",
                        "text": request.prompt
                    }
                ]
            }],
            "temperature": request.temperature.unwrap_or(0.7),
            "max_tokens": request.max_tokens.unwrap_or(512),
        });

        let url = format!("http://127.0.0.1:{}/v1/chat/completions", self.port);
        let client = reqwest::blocking::Client::new();
        let resp = client
            .post(&url)
            .json(&body)
            .timeout(std::time::Duration::from_secs(300))
            .send()
            .context("Failed to send VLM request")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            eyre::bail!("llama-server VLM returned {status}: {body}");
        }

        let chat_resp: serde_json::Value = resp.json()?;
        let content = chat_resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let prompt_tokens = chat_resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
        let completion_tokens = chat_resp["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;

        Ok(VlmCompletionResponse {
            id: format!("vlm-ascend-{}", uuid::Uuid::new_v4()),
            object: "vlm.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: request.model.clone(),
            content,
            usage: VlmUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        })
    }
}

impl Drop for AscendVlmServer {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.child.lock() {
            if let Some(ref mut child) = *guard {
                let _ = child.kill();
                let _ = child.wait();
                tracing::info!("llama-server (VLM, port {}) stopped", self.port);
            }
        }
    }
}

// ============================================================================
// OuteTTS Engine (alternative TTS via llama-tts)
// ============================================================================

/// Ascend-based OuteTTS engine using the llama-tts binary.
pub struct AscendOuteTtsEngine {
    config: AscendConfig,
}

impl AscendOuteTtsEngine {
    pub fn new(config: AscendConfig) -> Result<Self> {
        let model = config
            .outetts_model
            .as_ref()
            .ok_or_else(|| eyre::eyre!("ASCEND_OUTETTS_MODEL not set"))?;
        let vocoder = config
            .outetts_vocoder
            .as_ref()
            .ok_or_else(|| eyre::eyre!("ASCEND_OUTETTS_VOCODER not set"))?;
        if !model.exists() {
            eyre::bail!("OuteTTS model not found: {}", model.display());
        }
        if !vocoder.exists() {
            eyre::bail!("OuteTTS vocoder not found: {}", vocoder.display());
        }
        tracing::info!("Ascend OuteTTS ready: model={}", model.display());
        Ok(Self { config })
    }

    /// Synthesize speech from text using OuteTTS.
    pub fn synthesize(&self, text: &str) -> Result<Vec<u8>> {
        let model = self.config.outetts_model.as_ref().unwrap();
        let vocoder = self.config.outetts_vocoder.as_ref().unwrap();

        let tmp_output = tempfile::Builder::new()
            .suffix(".wav")
            .tempfile()
            .context("Failed to create temp output")?;

        let mut cmd = self.config.command("llama-tts");
        cmd.args([
            "-m",
            model.to_str().unwrap(),
            "-mv",
            vocoder.to_str().unwrap(),
            "-p",
            text,
            "-o",
            tmp_output.path().to_str().unwrap(),
            "-t",
            &self.config.threads.to_string(),
            "-ngl",
            &self.config.gpu_layers.to_string(),
        ]);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        tracing::info!("Running Ascend OuteTTS: text='{}'", text);
        let output = cmd.output().context("Failed to run llama-tts")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eyre::bail!("llama-tts failed: {}", stderr.lines().last().unwrap_or("unknown"));
        }

        let wav_data = std::fs::read(tmp_output.path()).context("Failed to read OuteTTS output")?;
        if wav_data.is_empty() {
            eyre::bail!("llama-tts produced empty output");
        }
        tracing::info!("Ascend OuteTTS complete: {} bytes WAV", wav_data.len());
        Ok(wav_data)
    }
}

// ============================================================================
// Image Engine
// ============================================================================

/// Ascend-based image generation using ominix-diffusion-cli binary.
pub struct AscendImageEngine {
    config: AscendConfig,
}

impl AscendImageEngine {
    pub fn new(config: AscendConfig) -> Result<Self> {
        let model = config
            .diffusion_model
            .as_ref()
            .ok_or_else(|| eyre::eyre!("ASCEND_DIFFUSION_MODEL not set"))?;
        if !model.exists() {
            eyre::bail!("Diffusion model not found: {}", model.display());
        }
        tracing::info!("Ascend Image ready: model={}", model.display());
        Ok(Self { config })
    }

    /// Generate an image from a text prompt.
    pub fn generate(&self, request: &ImageGenerationRequest) -> Result<ImageGenerationResponse> {
        let model = self.config.diffusion_model.as_ref().unwrap();
        let vae = self.config.diffusion_vae.as_ref()
            .ok_or_else(|| eyre::eyre!("ASCEND_DIFFUSION_VAE not set"))?;
        let llm = self.config.diffusion_llm.as_ref()
            .ok_or_else(|| eyre::eyre!("ASCEND_DIFFUSION_LLM not set"))?;

        let (width, height) = parse_size(&request.size);

        let tmp_output = tempfile::Builder::new()
            .suffix(".png")
            .tempfile()
            .context("Failed to create temp output file")?;

        let mut cmd = self.config.command("ominix-diffusion-cli");
        cmd.args([
            "--diffusion-model",
            model.to_str().unwrap(),
            "--vae",
            vae.to_str().unwrap(),
            "--llm",
            llm.to_str().unwrap(),
            "-p",
            &request.prompt,
            "--steps",
            "20",
            "--cfg-scale",
            "2.5",
            "--sampling-method",
            "euler",
            "--diffusion-fa",
            "--flow-shift",
            "3",
            "-W",
            &width.to_string(),
            "-H",
            &height.to_string(),
            "-o",
            tmp_output.path().to_str().unwrap(),
        ]);

        // Add init image for img2img / editing
        if let Some(ref init_img_b64) = request.image {
            let init_bytes = base64::Engine::decode(
                &base64::engine::general_purpose::STANDARD,
                init_img_b64,
            )
            .context("Invalid base64 in image field")?;
            let tmp_init = write_audio_tempfile(&init_bytes, ".png")?;
            cmd.args([
                "-i",
                tmp_init.path().to_str().unwrap(),
                "--strength",
                &request.strength.to_string(),
            ]);
        }

        // Enable CANN optimizations
        cmd.env("GGML_CANN_ACL_GRAPH", "1");
        cmd.env("GGML_CANN_QUANT_BF16", "on");

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        tracing::info!("Running Ascend diffusion: prompt='{}', {}x{}", request.prompt, width, height);
        let output = cmd.output().context("Failed to run ominix-diffusion-cli")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eyre::bail!(
                "ominix-diffusion-cli failed: {}",
                stderr.lines().last().unwrap_or("unknown")
            );
        }

        // Read generated image and base64 encode
        let png_data = std::fs::read(tmp_output.path())
            .context("Failed to read generated image")?;

        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&png_data);

        Ok(ImageGenerationResponse {
            created: chrono::Utc::now().timestamp(),
            data: vec![ImageData {
                url: None,
                b64_json: Some(b64),
                revised_prompt: None,
            }],
        })
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Find the best ASR decoder GGUF (prefer Q8_0 quantized).
fn find_asr_decoder(model_dir: &Path) -> Result<PathBuf> {
    let q8 = model_dir.join("qwen_asr_decoder_q8_0.gguf");
    if q8.exists() {
        return Ok(q8);
    }
    let f16 = model_dir.join("qwen_asr_decoder.gguf");
    if f16.exists() {
        return Ok(f16);
    }
    eyre::bail!(
        "No ASR decoder GGUF found in {}. Expected qwen_asr_decoder_q8_0.gguf or qwen_asr_decoder.gguf",
        model_dir.display()
    )
}

/// Parse ASR output text from qwen_asr stdout.
/// The text follows the "=== Result ===" marker.
fn parse_asr_output(stdout: &str) -> String {
    if let Some(idx) = stdout.find("=== Result ===") {
        let after = &stdout[idx + "=== Result ===".len()..];
        after.trim().to_string()
    } else {
        // Fallback: return last non-empty line
        stdout
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("")
            .trim()
            .to_string()
    }
}

/// Parse audio duration from ASR stderr timing output.
fn parse_asr_duration(stderr: &str) -> Option<f32> {
    // Look for pattern: "audio: NNNNN samples (X.XXs)"
    for line in stderr.lines() {
        if line.contains("samples") && line.contains("s)") {
            if let Some(start) = line.find('(') {
                if let Some(end) = line.find("s)") {
                    let duration_str = &line[start + 1..end];
                    return duration_str.parse().ok();
                }
            }
        }
    }
    None
}

/// Decode audio from request: local path or base64.
fn decode_request_audio(file: &str) -> Result<Vec<u8>> {
    if file.starts_with('/') {
        std::fs::read(file).with_context(|| format!("Failed to read audio: {}", file))
    } else {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD
            .decode(file)
            .context("Failed to decode base64 audio")
    }
}

/// Write audio bytes to a temp WAV file.
fn write_wav_tempfile(audio_bytes: &[u8]) -> Result<tempfile::NamedTempFile> {
    let suffix = if audio_bytes.len() >= 4 && &audio_bytes[..4] == b"RIFF" {
        ".wav"
    } else {
        // Not WAV — convert via ffmpeg first
        return write_and_convert_to_wav(audio_bytes);
    };
    let mut tmp = tempfile::Builder::new()
        .suffix(suffix)
        .tempfile()
        .context("Failed to create temp file")?;
    std::io::Write::write_all(&mut tmp, audio_bytes)?;
    Ok(tmp)
}

/// Convert arbitrary audio bytes to WAV via ffmpeg, write to temp file.
fn write_and_convert_to_wav(audio_bytes: &[u8]) -> Result<tempfile::NamedTempFile> {
    use std::io::Write;

    let tmp_out = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .context("Failed to create temp file")?;

    let mut child = Command::new("ffmpeg")
        .args([
            "-i",
            "pipe:0",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            tmp_out.path().to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("ffmpeg not found")?;

    child.stdin.take().unwrap().write_all(audio_bytes)?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        eyre::bail!("ffmpeg conversion failed");
    }
    Ok(tmp_out)
}

/// Write bytes to a temp file with given suffix.
fn write_audio_tempfile(bytes: &[u8], suffix: &str) -> Result<tempfile::NamedTempFile> {
    let mut tmp = tempfile::Builder::new()
        .suffix(suffix)
        .tempfile()
        .context("Failed to create temp file")?;
    std::io::Write::write_all(&mut tmp, bytes)?;
    Ok(tmp)
}

/// Parse "WIDTHxHEIGHT" size string (e.g., "1024x1024").
fn parse_size(size: &str) -> (u32, u32) {
    let parts: Vec<&str> = size.split('x').collect();
    if parts.len() == 2 {
        let w = parts[0].parse().unwrap_or(1024);
        let h = parts[1].parse().unwrap_or(1024);
        (w, h)
    } else {
        (1024, 1024)
    }
}
