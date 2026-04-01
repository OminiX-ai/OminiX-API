//! Qwen3-TTS engine with x-vector voice cloning support.
//!
//! Uses qwen3-tts-mlx for inference-time voice cloning: a 3-10s reference
//! audio clip is processed by ECAPA-TDNN to extract a speaker embedding,
//! which conditions the TTS generation to match the reference speaker.

use eyre::{Context, Result};
use qwen3_tts_mlx::{Synthesizer, SynthesizeOptions};

use crate::types::SpeechRequest;

/// Minimum characters before emitting a segment. Fragments shorter than this
/// are merged with the next to avoid tiny segments with poor prosody.
const MIN_SENTENCE_CHARS: usize = 10;

/// Estimate a safe `max_new_tokens` cap for a text segment.
///
/// At 12Hz codec rate, 1 frame ≈ 83ms of audio. Chinese text averages ~250ms
/// per character; English ~100ms per word. We use a generous 4x multiplier on
/// the estimated frames so the model has room to breathe (pauses, prosody)
/// but can't run away to 8192 frames for a short sentence.
/// Minimum 256 frames (~21s) to handle short text with long pauses.
pub fn max_tokens_for_text(text: &str) -> i32 {
    let char_count = text.chars().count();
    // ~3 codec frames per CJK character, ~1.5 per ASCII char, at 12Hz.
    // 4x headroom to avoid premature cutoff.
    let estimated_frames = (char_count as f32 * 3.0 * 4.0) as i32;
    estimated_frames.max(256).min(4096)
}

/// Split text at punctuation marks into segments for independent TTS synthesis.
///
/// Splits at all common punctuation: CJK (。！？…，；：、) and ASCII
/// (. ! ? , ; :) when followed by whitespace. Segments shorter than
/// `MIN_SENTENCE_CHARS` are merged with the next to avoid tiny fragments.
/// No force-splitting of long segments — only natural punctuation boundaries.
pub fn split_sentences(text: &str) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }

    let mut raw = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();

    for (idx, &ch) in chars.iter().enumerate() {
        current.push(ch);

        // CJK punctuation — always split
        let is_cjk_punct = matches!(ch, '。' | '！' | '？' | '…' | '，' | '；' | '：' | '、');

        // ASCII punctuation — split when followed by whitespace or end of text
        let is_ascii_punct = matches!(ch, '.' | '!' | '?' | ',' | ';' | ':') && {
            let next = chars.get(idx + 1);
            match next {
                None => true,
                Some(c) => c.is_whitespace(),
            }
        };

        let is_newline = ch == '\n';

        if is_cjk_punct || is_ascii_punct || is_newline {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                raw.push(trimmed);
            }
            current.clear();
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        raw.push(trimmed);
    }

    // Merge tiny segments with the next one
    let mut result = Vec::new();
    let mut carry = String::new();
    for s in raw {
        if !carry.is_empty() {
            carry.push_str(&s);
        } else {
            carry = s;
        }
        if carry.chars().count() >= MIN_SENTENCE_CHARS {
            result.push(carry.clone());
            carry.clear();
        }
    }
    if !carry.is_empty() {
        if let Some(last) = result.last_mut() {
            last.push_str(&carry);
        } else {
            result.push(carry);
        }
    }

    result
}

pub fn nonempty_instruct(instruct: Option<&str>) -> Option<&str> {
    instruct.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    })
}


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
        let instruct = nonempty_instruct(request.instruct.as_deref());

        let opts = SynthesizeOptions {
            speaker,
            language,
            speed_factor: if request.speed != 1.0 { Some(request.speed) } else { None },
            ..Default::default()
        };

        let samples = if let Some(instruct) = instruct {
            self.synthesizer
                .synthesize_with_speaker_instruct(&request.input, instruct, &opts)
                .map_err(|e| eyre::eyre!("Qwen3-TTS speaker+instruct synthesis failed: {e}"))?
        } else {
            self.synthesizer
                .synthesize(&request.input, &opts)
                .map_err(|e| eyre::eyre!("Qwen3-TTS synthesis failed: {e}"))?
        };

        self.samples_to_wav(&samples, self.synthesizer.sample_rate)
    }

    /// Synthesize speech using x-vector voice cloning.
    /// Loads reference audio from `ref_audio_path`, extracts speaker embedding,
    /// then generates speech in that voice.
    ///
    /// For long text, automatically splits at sentence boundaries and synthesizes
    /// each sentence independently (reusing the same speaker embedding). This
    /// avoids hitting the per-call token limit (~25-30s audio) of the Base model.
    pub fn synthesize_clone(
        &mut self,
        text: &str,
        ref_audio_path: &str,
        language: &str,
        speed: f32,
        instruct: Option<&str>,
    ) -> Result<Vec<u8>> {
        let expanded = crate::utils::expand_tilde(ref_audio_path);
        let instruct = nonempty_instruct(instruct);

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

        // Split into sentences for long text to avoid per-call token limits
        let sentences = split_sentences(text);

        let all_samples = if sentences.len() <= 1 {
            // Short text — single pass
            let opts = SynthesizeOptions {
                language,
                speed_factor: if speed != 1.0 { Some(speed) } else { None },
                max_new_tokens: Some(max_tokens_for_text(text)),
                ..Default::default()
            };
            if let Some(instruct) = instruct {
                self.synthesizer
                    .synthesize_voice_clone_instruct(text, &ref_samples, instruct, language, &opts)
                    .map_err(|e| eyre::eyre!("Voice clone+instruct failed: {e}"))?
            } else {
                self.synthesizer
                    .synthesize_voice_clone(text, &ref_samples, language, &opts)
                    .map_err(|e| eyre::eyre!("Voice cloning failed: {e}"))?
            }
        } else {
            // Long text — synthesize each sentence with the same speaker embedding
            tracing::info!(
                "Clone sentence chunking: {} sentences from {} chars",
                sentences.len(),
                text.chars().count()
            );

            let mut all = Vec::new();
            let mut skipped = 0usize;
            for (i, sentence) in sentences.iter().enumerate() {
                let opts = SynthesizeOptions {
                    language,
                    speed_factor: if speed != 1.0 { Some(speed) } else { None },
                    max_new_tokens: Some(max_tokens_for_text(sentence)),
                    ..Default::default()
                };
                let result = if let Some(instruct) = instruct {
                    self.synthesizer
                        .synthesize_voice_clone_instruct(
                            sentence,
                            &ref_samples,
                            instruct,
                            language,
                            &opts,
                        )
                } else {
                    self.synthesizer
                        .synthesize_voice_clone(sentence, &ref_samples, language, &opts)
                };

                let samples = match result {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(
                            "Clone sentence {i}/{} failed ({}chars), skipping: {e}",
                            sentences.len(),
                            sentence.chars().count(),
                        );

                        skipped += 1;
                        continue;
                    }
                };

                tracing::debug!(
                    "Clone sentence {}/{}: {:.1}s audio, {} chars",
                    i + 1,
                    sentences.len(),
                    samples.len() as f32 / self.synthesizer.sample_rate as f32,
                    sentence.chars().count()
                );

                all.extend_from_slice(&samples);
            }
            if skipped > 0 {
                tracing::warn!(
                    "Clone chunking: {skipped}/{} sentences skipped due to errors",
                    sentences.len()
                );
            }

            let duration = all.len() as f32 / self.synthesizer.sample_rate as f32;
            tracing::info!(
                "Clone sentence chunking complete: {:.1}s audio ({} samples)",
                duration,
                all.len()
            );
            all
        };


        self.samples_to_wav(&all_samples, self.synthesizer.sample_rate)
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

    /// Load and resample reference audio to 24kHz f32 samples.
    /// Call once per clone request; reuse the returned samples for each sentence.
    pub fn load_ref_audio(ref_audio_path: &str) -> Result<Vec<f32>> {
        let expanded = crate::utils::expand_tilde(ref_audio_path);
        if !std::path::Path::new(&expanded).exists() {
            return Err(eyre::eyre!("Reference audio not found: {expanded}"));
        }
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
        Ok(ref_samples)
    }

    /// Synthesize a single sentence with a preset speaker. Returns raw PCM i16 bytes.
    pub fn synthesize_one_sentence(
        &mut self,
        sentence: &str,
        speaker: &str,
        language: &str,
        speed: f32,
        instruct: Option<&str>,
    ) -> Result<Vec<u8>> {
        let instruct = nonempty_instruct(instruct);
        let opts = SynthesizeOptions {
            speaker,
            language,
            speed_factor: if speed != 1.0 { Some(speed) } else { None },
            max_new_tokens: Some(max_tokens_for_text(sentence)),
            ..Default::default()
        };
        let result = if let Some(instruct) = instruct {
            self.synthesizer
                .synthesize_with_speaker_instruct(sentence, instruct, &opts)
        } else {
            self.synthesizer.synthesize(sentence, &opts)
        };
        match result {
            Ok(samples) => {
                tracing::debug!(
                    "Single sentence: {:.1}s audio, {} chars",
                    samples.len() as f32 / self.synthesizer.sample_rate as f32,
                    sentence.chars().count()
                );
                Ok(Self::samples_to_pcm(&samples))
            }
            Err(e) => Err(eyre::eyre!("TTS sentence synthesis failed: {e}")),
        }
    }

    /// Synthesize a single sentence with voice cloning. Returns raw PCM i16 bytes.
    /// `ref_samples` should be pre-loaded via `load_ref_audio()`.
    pub fn synthesize_clone_one_sentence(
        &mut self,
        sentence: &str,
        ref_samples: &[f32],
        language: &str,
        speed: f32,
        instruct: Option<&str>,
    ) -> Result<Vec<u8>> {
        let instruct = nonempty_instruct(instruct);
        let opts = SynthesizeOptions {
            language,
            speed_factor: if speed != 1.0 { Some(speed) } else { None },
            max_new_tokens: Some(max_tokens_for_text(sentence)),
            ..Default::default()
        };
        let result = if let Some(instruct) = instruct {
            self.synthesizer
                .synthesize_voice_clone_instruct(sentence, ref_samples, instruct, language, &opts)
        } else {
            self.synthesizer
                .synthesize_voice_clone(sentence, ref_samples, language, &opts)
        };
        match result {
            Ok(samples) => {
                tracing::debug!(
                    "Clone single sentence: {:.1}s audio, {} chars",
                    samples.len() as f32 / self.synthesizer.sample_rate as f32,
                    sentence.chars().count()
                );
                Ok(Self::samples_to_pcm(&samples))
            }
            Err(e) => Err(eyre::eyre!("Clone sentence synthesis failed: {e}")),
        }
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
