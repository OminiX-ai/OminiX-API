//! ASR (Automatic Speech Recognition) engine
//!
//! Supports three backends:
//! - **Paraformer**: Fast CTC-based model (funasr-mlx)
//! - **SenseVoice + Qwen3-4B**: LLM-based multimodal ASR (funasr-qwen4b-mlx)
//! - **Qwen3-ASR**: Encoder-decoder ASR with 30+ languages (qwen3-asr-mlx)
//!
//! Backend is auto-detected from the model directory contents.

use std::path::{Path, PathBuf};

use eyre::{Context, Result};
use mlx_rs::module::Module;

use crate::model_config::{self, ModelAvailability, ModelCategory};
use crate::types::{TranscriptionRequest, TranscriptionResponse};

/// ASR backend type
enum AsrBackend {
    /// CTC-based Paraformer model
    Paraformer {
        model: funasr_mlx::Paraformer,
        vocab: funasr_mlx::Vocabulary,
    },
    /// LLM-based SenseVoice encoder + Qwen3-4B
    SenseVoiceQwen {
        model: funasr_qwen4b_mlx::FunASRQwen4B,
    },
    /// Qwen3-ASR encoder-decoder (0.6B / 1.7B, 30+ languages)
    Qwen3Asr {
        model: qwen3_asr_mlx::Qwen3ASR,
    },
}

/// ASR inference engine (auto-detects backend from model directory)
pub struct AsrEngine {
    backend: AsrBackend,
}

impl AsrEngine {
    /// Create a new ASR engine
    ///
    /// Auto-detects backend:
    /// - If config.json has `model_type == "qwen3_asr"` → Qwen3-ASR
    /// - If directory contains `adaptor*.safetensors` → SenseVoice + Qwen3-4B
    /// - Otherwise → Paraformer
    pub fn new(model_dir: &str) -> Result<Self> {
        // Resolve model directory
        let actual_model_dir = resolve_model_dir(model_dir)?;

        // Auto-detect backend
        if is_qwen3_asr_model(&actual_model_dir) {
            tracing::info!("Detected Qwen3-ASR model");
            Self::new_qwen3_asr(&actual_model_dir)
        } else if has_sensevoice_qwen_files(&actual_model_dir) {
            tracing::info!("Detected SenseVoice + Qwen3-4B ASR model");
            Self::new_sensevoice_qwen(&actual_model_dir)
        } else {
            tracing::info!("Detected Paraformer ASR model");
            Self::new_paraformer(&actual_model_dir)
        }
    }

    /// Load Paraformer backend
    fn new_paraformer(model_dir: &Path) -> Result<Self> {
        let weights_path = model_dir.join("paraformer.safetensors");
        let cmvn_path = model_dir.join("am.mvn");
        let vocab_path = model_dir.join("tokens.txt");

        if !weights_path.exists() {
            return Err(eyre::eyre!("ASR model weights not found at {:?}", weights_path));
        }
        if !cmvn_path.exists() {
            return Err(eyre::eyre!("ASR CMVN file not found at {:?}", cmvn_path));
        }
        if !vocab_path.exists() {
            return Err(eyre::eyre!("ASR vocabulary not found at {:?}", vocab_path));
        }

        let mut model = funasr_mlx::load_model(weights_path.to_str().unwrap())
            .context("Failed to load Paraformer model")?;
        model.training_mode(false);

        let (addshift, rescale) = funasr_mlx::parse_cmvn_file(cmvn_path.to_str().unwrap())
            .context("Failed to load CMVN file")?;
        model.set_cmvn(addshift, rescale);

        let vocab = funasr_mlx::Vocabulary::load(vocab_path.to_str().unwrap())
            .context("Failed to load vocabulary")?;

        tracing::info!("Paraformer ASR loaded: {} tokens", vocab.len());

        Ok(Self {
            backend: AsrBackend::Paraformer { model, vocab },
        })
    }

    /// Load SenseVoice + Qwen3-4B backend
    fn new_sensevoice_qwen(model_dir: &Path) -> Result<Self> {
        let model = funasr_qwen4b_mlx::FunASRQwen4B::load(
            model_dir.to_str().unwrap()
        ).map_err(|e| eyre::eyre!("Failed to load SenseVoice+Qwen4B: {:?}", e))?;

        tracing::info!("SenseVoice + Qwen3-4B ASR loaded");

        Ok(Self {
            backend: AsrBackend::SenseVoiceQwen { model },
        })
    }

    /// Load Qwen3-ASR backend
    fn new_qwen3_asr(model_dir: &Path) -> Result<Self> {
        let model = qwen3_asr_mlx::Qwen3ASR::load(model_dir)
            .map_err(|e| eyre::eyre!("Failed to load Qwen3-ASR: {:?}", e))?;

        tracing::info!("Qwen3-ASR loaded from {:?}", model_dir);

        Ok(Self {
            backend: AsrBackend::Qwen3Asr { model },
        })
    }

    /// Transcribe audio to text
    pub fn transcribe(&mut self, request: &TranscriptionRequest) -> Result<TranscriptionResponse> {
        let (samples, duration_secs) = decode_audio(request)?;

        match &mut self.backend {
            AsrBackend::Paraformer { model, vocab } => {
                let mut model = model.clone();
                let text = funasr_mlx::transcribe(&mut model, &samples, vocab)
                    .context("Paraformer transcription failed")?;

                Ok(TranscriptionResponse {
                    text,
                    language: request.language.clone(),
                    duration: Some(duration_secs),
                })
            }
            AsrBackend::SenseVoiceQwen { model } => {
                let text = model.transcribe_samples(&samples, 16000)
                    .map_err(|e| eyre::eyre!("SenseVoice+Qwen transcription failed: {:?}", e))?;

                Ok(TranscriptionResponse {
                    text,
                    language: request.language.clone().or(Some("zh".to_string())),
                    duration: Some(duration_secs),
                })
            }
            AsrBackend::Qwen3Asr { model } => {
                let language = request.language.as_deref().unwrap_or("Chinese");
                let text = model.transcribe_samples(&samples, language)
                    .map_err(|e| eyre::eyre!("Qwen3-ASR transcription failed: {:?}", e))?;

                Ok(TranscriptionResponse {
                    text,
                    language: Some(language.to_string()),
                    duration: Some(duration_secs),
                })
            }
        }
    }
}

/// Decode base64 audio from request, parse WAV, resample to 16kHz
fn decode_audio(request: &TranscriptionRequest) -> Result<(Vec<f32>, f32)> {
    let audio_bytes = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        &request.file,
    ).context("Failed to decode base64 audio")?;

    let cursor = std::io::Cursor::new(audio_bytes);
    let reader = hound::WavReader::new(cursor)
        .context("Failed to parse WAV file")?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => {
            reader
                .into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect()
        }
    };

    let duration_secs = samples.len() as f32 / sample_rate as f32;

    let samples = if sample_rate != 16000 {
        funasr_mlx::audio::resample(&samples, sample_rate, 16000)
    } else {
        samples
    };

    Ok((samples, duration_secs))
}

/// Check if directory is a Qwen3-ASR model (config.json with model_type "qwen3_asr")
fn is_qwen3_asr_model(model_dir: &Path) -> bool {
    let config_path = model_dir.join("config.json");
    if let Ok(content) = std::fs::read_to_string(&config_path) {
        if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
                return model_type == "qwen3_asr";
            }
        }
    }
    false
}

/// Check if directory contains SenseVoice + Qwen3-4B model files
fn has_sensevoice_qwen_files(model_dir: &Path) -> bool {
    // Look for adaptor weights (the distinguishing file)
    let has_adaptor = model_dir.join("adaptor.safetensors").exists()
        || model_dir.join("adaptor_phase2_final.safetensors").exists();

    if !has_adaptor {
        // Also check with glob-like pattern
        if let Ok(entries) = std::fs::read_dir(model_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("adaptor") && name.ends_with(".safetensors") {
                    return true;
                }
            }
        }
    }

    has_adaptor
}

/// Resolve model directory from config or path
fn resolve_model_dir(model_dir: &str) -> Result<PathBuf> {
    // Try known ASR models: qwen3-asr (best quality), funasr-qwen4b, funasr-paraformer
    for model_id in &["qwen3-asr", "qwen3-asr-1.7b", "qwen3-asr-0.6b", "funasr-qwen4b", "funasr-paraformer"] {
        match model_config::check_model(model_id, ModelCategory::Asr) {
            ModelAvailability::Ready { local_path, model_name } => {
                tracing::info!("Found locally available ASR model: {}", model_name);
                return Ok(local_path.unwrap_or_else(|| PathBuf::from(model_dir)));
            }
            _ => continue,
        }
    }

    // Fall back to hub cache or raw path
    if let Some(hub_path) = crate::utils::resolve_from_hub_cache(model_dir) {
        tracing::info!("Found ASR model in hub cache: {:?}", hub_path);
        let _ = model_config::register_model(model_dir, ModelCategory::Asr, &hub_path);
        Ok(hub_path)
    } else {
        Ok(PathBuf::from(model_dir))
    }
}
