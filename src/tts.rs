//! TTS (Text-to-Speech) engine using GPT-SoVITS

use eyre::{Context, Result};
use gpt_sovits_mlx::{VoiceCloner, VoiceClonerConfig};

use crate::types::SpeechRequest;

/// TTS inference engine using GPT-SoVITS
pub struct TtsEngine {
    cloner: VoiceCloner,
    ref_audio_path: String,
}

impl TtsEngine {
    /// Create a new TTS engine with a reference audio file
    pub fn new(ref_audio_path: &str) -> Result<Self> {
        // Check if reference audio exists
        if !std::path::Path::new(ref_audio_path).exists() {
            return Err(eyre::eyre!("Reference audio not found: {}", ref_audio_path));
        }

        // Initialize voice cloner
        let config = VoiceClonerConfig::default();
        let mut cloner = VoiceCloner::new(config)
            .context("Failed to initialize VoiceCloner")?;

        // Set reference audio (zero-shot mode)
        cloner.set_reference_audio(ref_audio_path)
            .context("Failed to set reference audio")?;

        tracing::info!("TTS engine initialized with reference: {}", ref_audio_path);

        Ok(Self {
            cloner,
            ref_audio_path: ref_audio_path.to_string(),
        })
    }

    /// Synthesize speech from text
    pub fn synthesize(&mut self, request: &SpeechRequest) -> Result<Vec<u8>> {
        // If a different voice is requested, try to use it as reference
        if let Some(ref voice) = request.voice {
            if voice != &self.ref_audio_path && std::path::Path::new(voice).exists() {
                self.cloner.set_reference_audio(voice)
                    .context("Failed to set new reference audio")?;
            }
        }

        // Synthesize
        let audio = self.cloner.synthesize(&request.input)
            .context("Synthesis failed")?;

        // Convert to WAV bytes
        let wav_bytes = self.samples_to_wav(&audio.samples, audio.sample_rate)?;

        Ok(wav_bytes)
    }

    /// Convert f32 samples to WAV bytes
    fn samples_to_wav(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        let cursor = std::io::Cursor::new(&mut buffer);

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::new(cursor, spec)
            .context("Failed to create WAV writer")?;

        for &sample in samples {
            let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(sample_i16)
                .context("Failed to write sample")?;
        }

        writer.finalize()
            .context("Failed to finalize WAV")?;

        Ok(buffer)
    }
}
