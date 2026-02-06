//! ASR (Automatic Speech Recognition) engine using funasr-mlx Paraformer

use std::path::PathBuf;

use eyre::{Context, Result};
use mlx_rs::module::Module;

use crate::model_config::{self, ModelAvailability, ModelCategory};
use crate::types::{TranscriptionRequest, TranscriptionResponse};

/// ASR inference engine using Paraformer
pub struct AsrEngine {
    model: funasr_mlx::Paraformer,
    vocab: funasr_mlx::Vocabulary,
}

impl AsrEngine {
    /// Create a new ASR engine
    ///
    /// model_dir should contain:
    /// - paraformer.safetensors
    /// - am.mvn (CMVN normalization)
    /// - tokens.txt (vocabulary)
    ///
    /// First checks ~/.OminiX/local_models_config.json for model availability.
    pub fn new(model_dir: &str) -> Result<Self> {
        // Check model configuration for local availability
        let actual_model_dir: PathBuf = match model_config::check_model("funasr-paraformer", ModelCategory::Asr) {
            ModelAvailability::Ready { local_path, model_name } => {
                tracing::info!("Found locally available ASR model: {}", model_name);
                local_path.unwrap_or_else(|| PathBuf::from(model_dir))
            }
            ModelAvailability::NotDownloaded { model_name, model_id } => {
                return Err(eyre::eyre!(
                    "ASR model '{}' ({}) is not downloaded.\n\
                     Please download it using OminiX-Studio before starting the API server.",
                    model_name, model_id
                ));
            }
            ModelAvailability::WrongCategory { .. } => {
                PathBuf::from(model_dir)
            }
            ModelAvailability::NotInConfig => {
                // Try standard model hub caches before falling back to raw path
                if let Some(hub_path) = crate::utils::resolve_from_hub_cache(model_dir) {
                    tracing::info!("Found ASR model in hub cache: {:?}", hub_path);
                    let _ = model_config::register_model(model_dir, ModelCategory::Asr, &hub_path);
                    hub_path
                } else {
                    PathBuf::from(model_dir)
                }
            }
        };

        let weights_path = actual_model_dir.join("paraformer.safetensors");
        let cmvn_path = actual_model_dir.join("am.mvn");
        let vocab_path = actual_model_dir.join("tokens.txt");

        // Check required files exist
        if !weights_path.exists() {
            return Err(eyre::eyre!("ASR model weights not found at {:?}", weights_path));
        }
        if !cmvn_path.exists() {
            return Err(eyre::eyre!("ASR CMVN file not found at {:?}", cmvn_path));
        }
        if !vocab_path.exists() {
            return Err(eyre::eyre!("ASR vocabulary not found at {:?}", vocab_path));
        }

        // Load model
        let mut model = funasr_mlx::load_model(weights_path.to_str().unwrap())
            .context("Failed to load Paraformer model")?;
        model.training_mode(false);

        // Load CMVN
        let (addshift, rescale) = funasr_mlx::parse_cmvn_file(cmvn_path.to_str().unwrap())
            .context("Failed to load CMVN file")?;
        model.set_cmvn(addshift, rescale);

        // Load vocabulary
        let vocab = funasr_mlx::Vocabulary::load(vocab_path.to_str().unwrap())
            .context("Failed to load vocabulary")?;

        tracing::info!("ASR model loaded: {} tokens", vocab.len());

        Ok(Self { model, vocab })
    }

    /// Transcribe audio to text
    pub fn transcribe(&self, request: &TranscriptionRequest) -> Result<TranscriptionResponse> {
        // Decode base64 audio data
        let audio_bytes = base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            &request.file,
        ).context("Failed to decode base64 audio")?;

        // Parse WAV file
        let cursor = std::io::Cursor::new(audio_bytes);
        let reader = hound::WavReader::new(cursor)
            .context("Failed to parse WAV file")?;

        let spec = reader.spec();
        let sample_rate = spec.sample_rate;

        // Read samples
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

        // Resample to 16kHz if needed
        let samples = if sample_rate != 16000 {
            funasr_mlx::audio::resample(&samples, sample_rate, 16000)
        } else {
            samples
        };

        // Transcribe
        let mut model = self.model.clone();
        let text = funasr_mlx::transcribe(&mut model, &samples, &self.vocab)
            .context("Transcription failed")?;

        Ok(TranscriptionResponse {
            text,
            language: request.language.clone(),
            duration: Some(duration_secs),
        })
    }
}
