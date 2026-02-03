//! TTS (Text-to-Speech) engine using GPT-SoVITS
//!
//! Supports:
//! - Voice registry via voices.json (named voices with aliases)
//! - Few-shot mode (reference audio + transcript for better quality)
//! - Pre-computed semantic codes (best quality)
//! - Zero-shot fallback (reference audio only)

use std::collections::BTreeMap;
use std::path::Path;
use eyre::{Context, Result};
use gpt_sovits_mlx::{VoiceCloner, VoiceClonerConfig};
use serde::Deserialize;

use crate::types::SpeechRequest;

// ============================================================================
// Voice Registry (voices.json)
// ============================================================================

const DEFAULT_VOICES_CONFIG: &str = "~/.dora/models/primespeech/voices.json";

#[derive(Debug, Deserialize)]
struct VoicesConfig {
    #[serde(default = "default_voice")]
    #[allow(dead_code)]
    default_voice: String,
    #[serde(default = "default_base_path")]
    models_base_path: String,
    voices: BTreeMap<String, VoiceConfig>,
}

#[derive(Debug, Deserialize, Clone)]
struct VoiceConfig {
    ref_audio: String,
    ref_text: String,
    #[serde(default)]
    codes_path: Option<String>,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    #[allow(dead_code)]
    speed_factor: Option<f32>,
    /// Finetuned VITS model weights (from voice cloning training)
    #[serde(default)]
    #[allow(dead_code)]
    vits_weights: Option<String>,
}

fn default_voice() -> String {
    "doubao".to_string()
}

fn default_base_path() -> String {
    "~/.dora/models/primespeech".to_string()
}

fn expand_tilde(path: &str) -> String {
    crate::utils::expand_tilde(path)
}

impl VoicesConfig {
    /// Load voice registry from VOICES_CONFIG env var or default path
    fn load() -> Option<Self> {
        let config_path = std::env::var("VOICES_CONFIG")
            .unwrap_or_else(|_| DEFAULT_VOICES_CONFIG.to_string());
        let config_path = expand_tilde(&config_path);

        let data = match std::fs::read_to_string(&config_path) {
            Ok(d) => d,
            Err(_) => {
                tracing::debug!("No voices.json found at {}", config_path);
                return None;
            }
        };

        match serde_json::from_str::<VoicesConfig>(&data) {
            Ok(config) => {
                tracing::info!("Loaded {} voices from {}", config.voices.len(), config_path);
                Some(config)
            }
            Err(e) => {
                tracing::warn!("Failed to parse voices.json: {}", e);
                None
            }
        }
    }

    /// Find a voice by name or alias (case-insensitive)
    fn find_voice(&self, name: &str) -> Option<&VoiceConfig> {
        let name_lower = name.to_lowercase();

        // Direct match
        if let Some(voice) = self.voices.get(&name_lower) {
            return Some(voice);
        }

        // Search aliases
        for voice in self.voices.values() {
            if voice.aliases.iter().any(|a| a.to_lowercase() == name_lower) {
                return Some(voice);
            }
        }

        None
    }

    /// Resolve a relative path against models_base_path
    fn resolve_path(&self, relative: &str) -> String {
        if Path::new(relative).is_absolute() {
            return relative.to_string();
        }
        let base = expand_tilde(&self.models_base_path);
        Path::new(&base).join(relative).to_string_lossy().to_string()
    }
}

// ============================================================================
// Path Security
// ============================================================================

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
            if let (Ok(voice_canonical), Ok(allowed_canonical)) = (
                path.canonicalize(),
                Path::new(&allowed_dir).canonicalize()
            ) {
                return voice_canonical.starts_with(&allowed_canonical);
            }
        }
        return false;
    }

    // Reject paths with traversal components
    for component in path.components() {
        match component {
            std::path::Component::ParentDir | std::path::Component::CurDir => return false,
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
    path.file_name()
        .map(|f| f.to_string_lossy() == voice)
        .unwrap_or(false)
}

// ============================================================================
// TTS Engine
// ============================================================================

/// TTS inference engine using GPT-SoVITS
pub struct TtsEngine {
    cloner: VoiceCloner,
    ref_audio_path: String,
    voices: Option<VoicesConfig>,
    current_voice: Option<String>,
}

impl TtsEngine {
    /// Create a new TTS engine with a reference audio file
    pub fn new(ref_audio_path: &str) -> Result<Self> {
        if !std::path::Path::new(ref_audio_path).exists() {
            return Err(eyre::eyre!("Reference audio not found: {}", ref_audio_path));
        }

        let config = VoiceClonerConfig::default();
        let mut cloner = VoiceCloner::new(config)
            .context("Failed to initialize VoiceCloner")?;

        // Set initial reference audio (zero-shot mode)
        cloner.set_reference_audio(ref_audio_path)
            .context("Failed to set reference audio")?;

        tracing::info!("TTS engine initialized with reference: {}", ref_audio_path);
        tracing::info!(
            "Few-shot mode: {}",
            if cloner.few_shot_available() { "available (HuBERT loaded)" } else { "unavailable (no HuBERT)" }
        );

        // Load voice registry
        let voices = VoicesConfig::load();

        Ok(Self {
            cloner,
            ref_audio_path: ref_audio_path.to_string(),
            voices,
            current_voice: None,
        })
    }

    /// Reload voice registry (called after training registers a new voice)
    pub fn reload_voices(&mut self) {
        self.voices = VoicesConfig::load();
        self.current_voice = None; // force re-evaluation on next request
        tracing::info!("Voice registry reloaded");
    }

    /// Synthesize speech from text
    pub fn synthesize(&mut self, request: &SpeechRequest) -> Result<Vec<u8>> {
        if let Some(ref voice) = request.voice {
            self.set_voice(voice)?;
        }

        let audio = self.cloner.synthesize(&request.input)
            .context("Synthesis failed")?;

        let wav_bytes = self.samples_to_wav(&audio.samples, audio.sample_rate)?;
        Ok(wav_bytes)
    }

    /// Switch to a named voice (from registry) or a file path
    fn set_voice(&mut self, voice_name: &str) -> Result<()> {
        // Skip if already using this voice
        if self.current_voice.as_deref() == Some(voice_name) {
            return Ok(());
        }

        // Try voice registry first — clone data out to avoid borrow conflicts
        let registry_match = self.voices.as_ref().and_then(|voices| {
            voices.find_voice(voice_name).map(|voice| {
                let ref_audio = voices.resolve_path(&voice.ref_audio);
                let ref_text = voice.ref_text.clone();
                let codes_path = voice.codes_path.as_ref().map(|c| voices.resolve_path(c));
                (ref_audio, ref_text, codes_path)
            })
        });

        if let Some((ref_audio, ref_text, codes_path)) = registry_match {
            if !Path::new(&ref_audio).exists() {
                return Err(eyre::eyre!(
                    "Voice '{}' reference audio not found: {}", voice_name, ref_audio
                ));
            }

            if let Some(codes) = codes_path {
                if Path::new(&codes).exists() {
                    // Best quality: pre-computed semantic codes
                    tracing::info!("Voice '{}': few-shot with pre-computed codes", voice_name);
                    self.cloner
                        .set_reference_with_precomputed_codes(&ref_audio, &ref_text, &codes)
                        .context("Failed to set reference with pre-computed codes")?;
                } else {
                    tracing::warn!("Voice '{}': codes file not found ({}), falling back", voice_name, codes);
                    self.set_voice_few_shot_or_zero(&ref_audio, &ref_text, voice_name)?;
                }
            } else {
                self.set_voice_few_shot_or_zero(&ref_audio, &ref_text, voice_name)?;
            }

            self.current_voice = Some(voice_name.to_string());
            return Ok(());
        }

        // Not in registry — treat as file path (existing behavior)
        if !is_safe_voice_path(voice_name) {
            return Err(eyre::eyre!(
                "Voice '{}' not found in registry and not a valid file path", voice_name
            ));
        }

        let voice_path = if let Some(allowed_dir) = allowed_voices_dir() {
            if Path::new(voice_name).is_absolute() {
                voice_name.to_string()
            } else {
                Path::new(&allowed_dir).join(voice_name).to_string_lossy().to_string()
            }
        } else {
            voice_name.to_string()
        };

        if voice_path != self.ref_audio_path && Path::new(&voice_path).exists() {
            tracing::info!("Voice '{}': zero-shot from file path", voice_name);
            self.cloner.set_reference_audio(&voice_path)
                .context("Failed to set reference audio")?;
        }

        self.current_voice = Some(voice_name.to_string());
        Ok(())
    }

    /// Try few-shot mode (HuBERT), fall back to zero-shot
    fn set_voice_few_shot_or_zero(&mut self, ref_audio: &str, ref_text: &str, voice_name: &str) -> Result<()> {
        if self.cloner.few_shot_available() {
            tracing::info!("Voice '{}': few-shot with HuBERT extraction", voice_name);
            self.cloner
                .set_reference_audio_with_text(ref_audio, ref_text)
                .context("Failed to set reference with text (few-shot)")?;
        } else {
            tracing::info!("Voice '{}': zero-shot (HuBERT unavailable)", voice_name);
            self.cloner.set_reference_audio(ref_audio)
                .context("Failed to set reference audio")?;
        }
        Ok(())
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
