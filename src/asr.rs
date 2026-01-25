//! ASR (Automatic Speech Recognition) engine using funasr-mlx Paraformer

use eyre::{Context, Result};
use mlx_rs::module::Module;

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
    pub fn new(model_dir: &str) -> Result<Self> {
        let weights_path = format!("{}/paraformer.safetensors", model_dir);
        let cmvn_path = format!("{}/am.mvn", model_dir);
        let vocab_path = format!("{}/tokens.txt", model_dir);

        // Load model
        let mut model = funasr_mlx::load_model(&weights_path)
            .context("Failed to load Paraformer model")?;
        model.training_mode(false);

        // Load CMVN
        let (addshift, rescale) = funasr_mlx::parse_cmvn_file(&cmvn_path)
            .context("Failed to load CMVN file")?;
        model.set_cmvn(addshift, rescale);

        // Load vocabulary
        let vocab = funasr_mlx::Vocabulary::load(&vocab_path)
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
