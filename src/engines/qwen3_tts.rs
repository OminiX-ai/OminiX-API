//! Qwen3-TTS engine with x-vector voice cloning support.
//!
//! Uses qwen3-tts-mlx for inference-time voice cloning: a 3-10s reference
//! audio clip is processed by ECAPA-TDNN to extract a speaker embedding,
//! which conditions the TTS generation to match the reference speaker.

use eyre::{Context, Result};
use qwen3_tts_mlx::{Synthesizer, SynthesizeOptions};

use crate::types::SpeechRequest;

/// Maximum characters (Unicode) per sentence chunk. Sentences longer than this
/// are force-split at clause boundaries or whitespace.
const MAX_SENTENCE_CHARS: usize = 200;

/// Minimum characters to accumulate before emitting a sentence. Tiny fragments
/// like "Yes." or "OK!" are merged with the next sentence to reduce per-call
/// overhead and improve prosody continuity.
const MIN_SENTENCE_CHARS: usize = 20;

/// Estimate a safe `max_new_tokens` cap for a text segment.
///
/// At 12Hz codec rate, 1 frame ≈ 83ms of audio. Chinese text averages ~250ms
/// per character; English ~100ms per word. We use a generous 4x multiplier on
/// the estimated frames so the model has room to breathe (pauses, prosody)
/// but can't run away to 8192 frames for a short sentence.
/// Minimum 256 frames (~21s) to handle short text with long pauses.
fn max_tokens_for_text(text: &str) -> i32 {
    let char_count = text.chars().count();
    // ~3 codec frames per CJK character, ~1.5 per ASCII char, at 12Hz.
    // 4x headroom to avoid premature cutoff.
    let estimated_frames = (char_count as f32 * 3.0 * 4.0) as i32;
    estimated_frames.max(256).min(4096)
}

/// Split text into sentences suitable for independent TTS synthesis.
///
/// Uses CJK sentence-ending punctuation (。！？) unconditionally. For ASCII
/// period, only splits when followed by whitespace + uppercase (avoids breaking
/// on abbreviations like "Dr." or decimals like "3.14"). Tiny sentences are
/// merged with the next. Oversized sentences are further split at clause
/// boundaries or whitespace.
fn split_sentences(text: &str) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }

    let mut raw_sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();

    for (idx, &ch) in chars.iter().enumerate() {
        current.push(ch);

        // CJK sentence enders — always split
        let is_cjk_end = matches!(ch, '。' | '！' | '？' | '…');

        // ASCII sentence enders — only split when followed by space+uppercase
        // or end of text, to avoid breaking "Dr.", "3.14", "U.S." etc.
        let is_ascii_end = matches!(ch, '!' | '?') || (ch == '.' && {
            // Check if next non-space char is uppercase or end of text
            let rest = &chars[idx + 1..];
            let next_alpha = rest.iter().skip_while(|c| c.is_whitespace()).next();
            match next_alpha {
                None => true,                   // end of text
                Some(c) => c.is_uppercase(),    // "word. Next" → split
            }
        });

        let is_newline = ch == '\n';

        if is_cjk_end || is_ascii_end || is_newline {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                raw_sentences.push(trimmed);
            }
            current.clear();
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        raw_sentences.push(trimmed);
    }

    // Merge tiny sentences with the next one
    let mut merged = Vec::new();
    let mut carry = String::new();
    for s in raw_sentences {
        if !carry.is_empty() {
            carry.push(' ');
            carry.push_str(&s);
        } else {
            carry = s;
        }
        if carry.chars().count() >= MIN_SENTENCE_CHARS {
            merged.push(carry.clone());
            carry.clear();
        }
    }
    if !carry.is_empty() {
        if let Some(last) = merged.last_mut() {
            last.push(' ');
            last.push_str(&carry);
        } else {
            merged.push(carry);
        }
    }

    // Split oversized sentences at clause boundaries
    let mut result = Vec::new();
    for sentence in merged {
        if sentence.chars().count() <= MAX_SENTENCE_CHARS {
            result.push(sentence);
        } else {
            split_long_sentence(&sentence, &mut result);
        }
    }

    result
}

fn nonempty_instruct(instruct: Option<&str>) -> Option<&str> {
    instruct.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    })
}

/// Split an oversized sentence at commas/semicolons, then whitespace.
fn split_long_sentence(text: &str, out: &mut Vec<String>) {
    let mut current = String::new();
    let mut char_count = 0usize;

    for ch in text.chars() {
        current.push(ch);
        char_count += 1;

        let is_clause_break = matches!(ch,
            ',' | ';' | ':' | '，' | '；' | '：' | '、'
        );

        if is_clause_break && char_count >= 30 {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                out.push(trimmed);
            }
            current.clear();
            char_count = 0;
        }
    }

    // If still oversized, split at whitespace
    let remaining = current.trim().to_string();
    if remaining.chars().count() <= MAX_SENTENCE_CHARS || !remaining.contains(' ') {
        if !remaining.is_empty() {
            out.push(remaining);
        }
    } else {
        let mut chunk = String::new();
        let mut chunk_chars = 0usize;
        for word in remaining.split_whitespace() {
            let word_chars = word.chars().count();
            if chunk_chars > 0 && chunk_chars + 1 + word_chars > MAX_SENTENCE_CHARS {
                out.push(chunk.clone());
                chunk.clear();
                chunk_chars = 0;
            }
            if chunk_chars > 0 {
                chunk.push(' ');
                chunk_chars += 1;
            }
            chunk.push_str(word);
            chunk_chars += word_chars;
        }
        if !chunk.is_empty() {
            out.push(chunk);
        }
    }
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

    /// Sentence-level streaming for voice cloning: split text, synthesize each
    /// sentence with the same speaker embedding, send PCM per sentence via callback.
    pub fn synthesize_clone_sentences(
        &mut self,
        text: &str,
        ref_audio_path: &str,
        language: &str,
        speed: f32,
        instruct: Option<&str>,
        mut on_pcm: impl FnMut(&[u8]) -> bool,
    ) -> Result<()> {
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

        let sentences = split_sentences(text);
        tracing::info!(
            "Clone sentence streaming: {} sentences from {} chars",
            sentences.len(),
            text.chars().count()
        );

        let mut total_samples = 0usize;
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
                    // Clear GPU cache to recover memory after OOM
            
                    skipped += 1;
                    continue;
                }
            };

            total_samples += samples.len();
            let pcm = Self::samples_to_pcm(&samples);

            tracing::debug!(
                "Clone sentence {}/{}: {:.1}s audio, {} chars",
                i + 1,
                sentences.len(),
                samples.len() as f32 / self.synthesizer.sample_rate as f32,
                sentence.chars().count()
            );

            if !on_pcm(&pcm) {
                tracing::info!("Client disconnected after clone sentence {i}");
                break;
            }
        }
        if skipped > 0 {
            tracing::warn!(
                "Clone streaming: {skipped}/{} sentences skipped due to errors",
                sentences.len()
            );
        }

        let duration = total_samples as f32 / self.synthesizer.sample_rate as f32;
        tracing::info!(
            "Clone sentence streaming complete: {:.1}s audio ({total_samples} samples)",
            duration
        );
        // Flush GPU cache after each request to prevent memory buildup across requests.

        Ok(())
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

    /// Sentence-level streaming: split text at sentence boundaries, synthesize
    /// each sentence independently, and send PCM via callback as each completes.
    ///
    /// This gives pseudo-streaming for long text — the client receives audio for
    /// the first sentence while later sentences are still generating. Each sentence
    /// is decoded in a single pass (no chunk boundary artifacts).
    pub fn synthesize_sentences(
        &mut self,
        request: &SpeechRequest,
        mut on_pcm: impl FnMut(&[u8]) -> bool, // return false to stop
    ) -> Result<()> {
        let speaker = request.voice.as_deref().unwrap_or("vivian");
        let language = request.language.as_deref().unwrap_or("chinese");
        let instruct = nonempty_instruct(request.instruct.as_deref());

        let sentences = split_sentences(&request.input);
        tracing::info!(
            "Sentence-level streaming: {} sentences from {} chars",
            sentences.len(),
            request.input.len()
        );

        let mut total_samples = 0usize;
        let mut skipped = 0usize;
        for (i, sentence) in sentences.iter().enumerate() {
            let opts = SynthesizeOptions {
                speaker,
                language,
                speed_factor: if request.speed != 1.0 { Some(request.speed) } else { None },
                max_new_tokens: Some(max_tokens_for_text(sentence)),
                ..Default::default()
            };
            let result = if let Some(instruct) = instruct {
                self.synthesizer
                    .synthesize_with_speaker_instruct(sentence, instruct, &opts)
            } else {
                self.synthesizer
                    .synthesize(sentence, &opts)
            };

            let samples = match result {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(
                        "TTS sentence {i}/{} failed ({}chars), skipping: {e}",
                        sentences.len(),
                        sentence.chars().count(),
                    );
            
                    skipped += 1;
                    continue;
                }
            };

            total_samples += samples.len();
            let pcm = Self::samples_to_pcm(&samples);

            tracing::debug!(
                "Sentence {i}/{}: {:.1}s audio, {} chars",
                sentences.len(),
                samples.len() as f32 / self.synthesizer.sample_rate as f32,
                sentence.len()
            );

            if !on_pcm(&pcm) {
                tracing::info!("Client disconnected after sentence {i}");
                break;
            }
        }
        if skipped > 0 {
            tracing::warn!(
                "TTS streaming: {skipped}/{} sentences skipped due to errors",
                sentences.len()
            );
        }

        let duration = total_samples as f32 / self.synthesizer.sample_rate as f32;
        tracing::info!("Sentence streaming complete: {:.1}s audio ({total_samples} samples)", duration);

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
