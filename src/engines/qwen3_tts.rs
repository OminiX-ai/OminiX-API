//! Qwen3-TTS engine with x-vector voice cloning support.
//!
//! Uses qwen3-tts-mlx for inference-time voice cloning: a 3-10s reference
//! audio clip is processed by ECAPA-TDNN to extract a speaker embedding,
//! which conditions the TTS generation to match the reference speaker.

use eyre::{Context, Result};
use qwen3_tts_mlx::{Synthesizer, SynthesizeOptions, DEFAULT_CHUNK_FRAMES};

use crate::types::SpeechRequest;

/// Qwen3-TTS inference engine.
pub struct Qwen3TtsEngine {
    synthesizer: Synthesizer,
}

impl Qwen3TtsEngine {
    /// Load the Qwen3-TTS model from a directory.
    /// Use a Base model for voice cloning (includes ECAPA-TDNN speaker encoder).
    pub fn new(model_dir: &str) -> Result<Self> {
        let expanded = crate::utils::expand_tilde(model_dir);
        let synthesizer = Synthesizer::load(&expanded)
            .map_err(|e| eyre::eyre!("Failed to load Qwen3-TTS: {e}"))?;

        let model_type = synthesizer.model_type();
        tracing::info!("Qwen3-TTS loaded (type: {model_type})");

        if synthesizer.speaker_encoder.is_some() {
            tracing::info!("Speaker encoder available — voice cloning enabled");
        } else {
            tracing::warn!("No speaker encoder — voice cloning unavailable (use Base model)");
        }

        Ok(Self { synthesizer })
    }

    /// Synthesize speech from text using a preset speaker.
    pub fn synthesize(&mut self, request: &SpeechRequest) -> Result<Vec<u8>> {
        let speaker = request.voice.as_deref().unwrap_or("vivian");
        let language = request.language.as_deref().unwrap_or("chinese");

        let opts = SynthesizeOptions {
            speaker,
            language,
            speed_factor: if request.speed != 1.0 { Some(request.speed) } else { None },
            ..Default::default()
        };

        let samples = self.synthesizer.synthesize(&request.input, &opts)
            .map_err(|e| eyre::eyre!("Qwen3-TTS synthesis failed: {e}"))?;

        self.samples_to_wav(&samples, self.synthesizer.sample_rate)
    }

    /// Synthesize speech using x-vector voice cloning.
    /// Loads reference audio from `ref_audio_path`, extracts speaker embedding,
    /// then generates speech in that voice.
    pub fn synthesize_clone(
        &mut self,
        text: &str,
        ref_audio_path: &str,
        language: &str,
        speed: f32,
    ) -> Result<Vec<u8>> {
        let expanded = crate::utils::expand_tilde(ref_audio_path);

        if !std::path::Path::new(&expanded).exists() {
            return Err(eyre::eyre!("Reference audio not found: {expanded}"));
        }

        // Load and resample to 24kHz
        let (samples, sr) = mlx_rs_core::audio::load_wav(&expanded)
            .context("Failed to load reference audio")?;

        let ref_samples = if sr != 24000 {
            tracing::info!("Resampling reference audio from {sr}Hz to 24000Hz");
            mlx_rs_core::audio::resample(&samples, sr, 24000)
        } else {
            samples
        };

        let duration_secs = ref_samples.len() as f32 / 24000.0;
        tracing::info!("Reference audio: {duration_secs:.1}s ({} samples)", ref_samples.len());

        let opts = SynthesizeOptions {
            language,
            speed_factor: if speed != 1.0 { Some(speed) } else { None },
            ..Default::default()
        };

        let output = self.synthesizer.synthesize_voice_clone(text, &ref_samples, language, &opts)
            .map_err(|e| eyre::eyre!("Voice cloning failed: {e}"))?;

        self.samples_to_wav(&output, self.synthesizer.sample_rate)
    }

    /// Whether voice cloning is supported (requires Base model with speaker encoder).
    pub fn supports_voice_cloning(&self) -> bool {
        self.synthesizer.supports_voice_cloning()
    }

    /// Whether preset speakers (vivian, ryan, etc.) are supported (requires CustomVoice model).
    pub fn supports_preset_speakers(&self) -> bool {
        self.synthesizer.supports_preset_speakers()
    }

    /// Sample rate of the loaded model (typically 24000).
    pub fn sample_rate(&self) -> u32 {
        self.synthesizer.sample_rate
    }

    /// Streaming synthesis: yields audio sample chunks via a callback.
    /// Each chunk is ~800ms of audio (DEFAULT_CHUNK_FRAMES=10 frames at 12Hz).
    pub fn synthesize_streaming(
        &mut self,
        request: &SpeechRequest,
        mut on_chunk: impl FnMut(Vec<f32>) -> bool, // return false to stop
    ) -> Result<()> {
        let speaker = request.voice.as_deref().unwrap_or("vivian");
        let language = request.language.as_deref().unwrap_or("chinese");

        let opts = SynthesizeOptions {
            speaker,
            language,
            speed_factor: if request.speed != 1.0 { Some(request.speed) } else { None },
            ..Default::default()
        };

        let mut session = self.synthesizer.start_streaming(&request.input, &opts, DEFAULT_CHUNK_FRAMES)
            .map_err(|e| eyre::eyre!("Failed to start streaming TTS: {e}"))?;

        while let Some(samples) = session.next_chunk()
            .map_err(|e| eyre::eyre!("Streaming TTS chunk error: {e}"))?
        {
            if !on_chunk(samples) {
                break;
            }
        }

        tracing::info!(
            "Streaming TTS complete: {:.1}s audio ({} samples)",
            session.duration_secs(),
            session.total_samples()
        );
        Ok(())
    }

    /// Convert f32 samples to raw PCM i16 bytes (no header, for streaming).
    pub fn samples_to_pcm(samples: &[f32]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            let i = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
            buf.extend_from_slice(&i.to_le_bytes());
        }
        buf
    }

    /// Convert f32 samples to WAV bytes (16-bit PCM, mono).
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

        writer.finalize().context("Failed to finalize WAV")?;
        Ok(buffer)
    }
}
