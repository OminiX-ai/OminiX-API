//! TTS (Text-to-Speech) engine using GPT-SoVITS

use std::path::Path;
use eyre::{Context, Result};
use gpt_sovits_mlx::{VoiceCloner, VoiceClonerConfig};

use crate::types::SpeechRequest;

/// Allowed directory for voice reference files (configurable via env)
fn allowed_voices_dir() -> Option<String> {
    std::env::var("TTS_VOICES_DIR").ok()
}

/// Validate that a voice path is safe (no path traversal)
fn is_safe_voice_path(voice: &str) -> bool {
    let path = Path::new(voice);

    // Reject absolute paths unless they're in the allowed directory
    if path.is_absolute() {
        if let Some(allowed_dir) = allowed_voices_dir() {
            // Canonicalize both paths to resolve any symlinks/.. sequences
            if let (Ok(voice_canonical), Ok(allowed_canonical)) = (
                path.canonicalize(),
                Path::new(&allowed_dir).canonicalize()
            ) {
                return voice_canonical.starts_with(&allowed_canonical);
            }
        }
        return false;
    }

    // Reject paths with traversal sequences
    let voice_str = voice.to_lowercase();
    if voice_str.contains("..") || voice_str.contains("./") || voice_str.contains("/.") {
        return false;
    }

    // Reject paths that look like they're trying to escape
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => return false,
            std::path::Component::CurDir => return false,
            _ => {}
        }
    }

    // If we have an allowed directory, construct the full path and validate
    if let Some(allowed_dir) = allowed_voices_dir() {
        let full_path = Path::new(&allowed_dir).join(voice);
        if let Ok(canonical) = full_path.canonicalize() {
            if let Ok(allowed_canonical) = Path::new(&allowed_dir).canonicalize() {
                return canonical.starts_with(&allowed_canonical);
            }
        }
        return false;
    }

    // Without an allowed directory configured, only accept simple filenames
    // (no directory components at all)
    path.file_name()
        .map(|f| f.to_string_lossy() == voice)
        .unwrap_or(false)
}

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
            // Security: Validate voice path to prevent path traversal attacks
            if !is_safe_voice_path(voice) {
                return Err(eyre::eyre!(
                    "Invalid voice path: must be a simple filename or within TTS_VOICES_DIR"
                ));
            }

            // Resolve the full path if using allowed directory
            let voice_path = if let Some(allowed_dir) = allowed_voices_dir() {
                if Path::new(voice).is_absolute() {
                    voice.to_string()
                } else {
                    Path::new(&allowed_dir).join(voice).to_string_lossy().to_string()
                }
            } else {
                voice.to_string()
            };

            if voice_path != self.ref_audio_path && Path::new(&voice_path).exists() {
                self.cloner.set_reference_audio(&voice_path)
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
