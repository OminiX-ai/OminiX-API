//! Qwen3-TTS engine management and request types.
//!
//! TTS requests are handled inline by the inference thread (not a separate pool)
//! to serialize all GPU access through a single thread. This prevents Metal
//! command buffer crashes when ASR and TTS run concurrently on separate threads.
//!
//! Only one model variant (CustomVoice or Base) is loaded at a time. When a
//! request requires the other variant, the engine swaps the talker weights
//! in-place — reusing the shared decoder and tokenizer to cut swap time.
//!
//! See: <https://github.com/ml-explore/mlx/issues/3078> (MLX thread safety)

use tokio::sync::oneshot;

use crate::engines::qwen3_tts;
use crate::model_config::{self, ModelAvailability, ModelCategory};
use crate::types::{SpeechCloneRequest, SpeechRequest};


/// Write raw audio bytes to a temp file for engine consumption.
/// The caller must keep the `NamedTempFile` alive until inference completes.
fn ref_audio_to_tempfile(bytes: &[u8]) -> eyre::Result<tempfile::NamedTempFile> {
    if bytes.len() < 44 {
        return Err(eyre::eyre!("Reference audio too small ({} bytes)", bytes.len()));
    }
    if bytes.len() > 10_000_000 {
        return Err(eyre::eyre!("Reference audio too large (>10MB)"));
    }
    let mut tmp = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .map_err(|e| eyre::eyre!("Failed to create temp file: {e}"))?;
    std::io::Write::write_all(&mut tmp, bytes)
        .map_err(|e| eyre::eyre!("Failed to write temp audio: {e}"))?;
    Ok(tmp)
}

// ── Public types ────────────────────────────────────────────────────

/// Request routed to the TTS pool (instead of the main inference thread).
pub enum TtsRequest {
    /// Non-streaming preset/legacy speech (returns complete WAV).
    Speech {
        request: SpeechRequest,
        response_tx: oneshot::Sender<eyre::Result<Vec<u8>>>,
    },
    /// Voice cloning (returns complete WAV).
    SpeechClone {
        request: SpeechCloneRequest,
        response_tx: oneshot::Sender<eyre::Result<Vec<u8>>>,
    },
    /// Synthesize a single sentence with a preset speaker. Returns PCM bytes.
    /// Used by per-sentence scheduling to avoid blocking the queue for all sentences.
    SpeechOneSentence {
        sentence: String,
        voice: String,
        language: String,
        speed: f32,
        instruct: Option<String>,
        response_tx: oneshot::Sender<eyre::Result<Vec<u8>>>,
    },
    /// Prepare reference audio for voice cloning (load + resample + cache).
    /// Must be called before `CloneOneSentence` requests.
    PrepareCloneRef {
        audio_bytes: Vec<u8>,
        response_tx: oneshot::Sender<eyre::Result<()>>,
    },
    /// Synthesize a single sentence using cached clone reference. Returns PCM bytes.
    CloneOneSentence {
        sentence: String,
        language: String,
        speed: f32,
        instruct: Option<String>,
        response_tx: oneshot::Sender<eyre::Result<Vec<u8>>>,
    },
}

/// Configuration for the TTS worker pool.
#[derive(Debug, Clone)]
pub struct TtsPoolConfig {
    /// When true (default), load the default model (CustomVoice) at startup
    /// instead of on first request. Set `TTS_LAZY_LOAD=1` to use lazy loading.
    pub eager_load: bool,
}

impl Default for TtsPoolConfig {
    fn default() -> Self {
        Self { eager_load: true }
    }
}

impl TtsPoolConfig {
    pub fn from_env() -> Self {
        let eager_load = std::env::var("TTS_LAZY_LOAD")
            .map(|v| v != "1" && v.to_lowercase() != "true")
            .unwrap_or(true);
        Self { eager_load }
    }
}

// ── Which variant is loaded ───────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TtsVariant {
    CustomVoice,
    Base,
}

impl TtsVariant {
    fn from_model_ref(model_ref: &str) -> Self {
        if model_ref.to_lowercase().contains("base") {
            Self::Base
        } else {
            Self::CustomVoice
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::CustomVoice => "customvoice",
            Self::Base => "base",
        }
    }
}

impl std::fmt::Display for TtsVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ── Model discovery ────────────────────────────────────────────────

fn has_qwen3_tts_weights(path: &std::path::Path) -> bool {
    path.join("model.safetensors").is_file()
        || path.join("model.safetensors.index.json").is_file()
}

fn read_tts_config(path: &std::path::Path) -> Option<serde_json::Value> {
    let content = std::fs::read_to_string(path.join("config.json")).ok()?;
    serde_json::from_str(&content).ok()
}

fn detect_tts_variant(path: &std::path::Path) -> Option<TtsVariant> {
    if let Some(config) = read_tts_config(path) {
        let model_type = config
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if model_type != "qwen3_tts" {
            return None;
        }

        let tts_model_type = config
            .get("tts_model_type")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_lowercase();
        return match tts_model_type.as_str() {
            "base" => Some(TtsVariant::Base),
            "customvoice" | "custom_voice" => Some(TtsVariant::CustomVoice),
            _ => None,
        };
    }

    let name = path.file_name()?.to_string_lossy().to_lowercase();
    if name.contains("base") {
        Some(TtsVariant::Base)
    } else if name.contains("customvoice") || name.contains("custom_voice") {
        Some(TtsVariant::CustomVoice)
    } else {
        None
    }
}

fn detect_quant_bits(path: &std::path::Path) -> Option<i64> {
    let config = read_tts_config(path)?;
    config
        .pointer("/quantization/bits")
        .or_else(|| config.pointer("/quantization_config/bits"))
        .and_then(|v| v.as_i64())
}

fn path_matches_variant(path: &std::path::Path, variant: TtsVariant) -> bool {
    has_qwen3_tts_weights(path) && detect_tts_variant(path) == Some(variant)
}

fn quant_bits_from_label(label: &str) -> Option<i64> {
    let lower = label.to_lowercase();
    (2..=8)
        .find(|bits| {
            lower.contains(&format!("{bits}bit")) || lower.contains(&format!("{bits}-bit"))
        })
        .map(i64::from)
}

fn catalog_quant_bits(model_ref: &str) -> Option<i64> {
    crate::model_registry::get_default_models()
        .into_iter()
        .find(|model| {
            model.category == ModelCategory::Tts
                && (model.id == model_ref || model.source.repo_id.as_deref() == Some(model_ref))
        })
        .and_then(|model| model.runtime.quantization.as_deref().and_then(quant_bits_from_label))
}

fn requested_quant_bits(model_ref: &str) -> Option<i64> {
    quant_bits_from_label(model_ref).or_else(|| catalog_quant_bits(model_ref))
}

fn path_matches_model_ref(path: &std::path::Path, model_ref: &str) -> bool {
    let lower = model_ref.to_lowercase();
    let path_lower = path.to_string_lossy().to_lowercase();

    if let Some(bits) = requested_quant_bits(model_ref) {
        return detect_quant_bits(path) == Some(bits)
            || path_lower.contains(&format!("{bits}bit"))
            || path_lower.contains(&format!("{bits}-bit"));
    }

    let leaf = lower.rsplit('/').next().unwrap_or(lower.as_str());
    let normalized_leaf = leaf.replace(['-', '_'], "");
    let normalized_path = path_lower.replace(['-', '_'], "");
    path_lower.contains(leaf) || normalized_path.contains(&normalized_leaf)
}

fn path_matches_requested_model(
    path: &std::path::Path,
    model_ref: &str,
    variant: TtsVariant,
) -> bool {
    path_matches_variant(path, variant) && path_matches_model_ref(path, model_ref)
}

fn collect_tts_candidates() -> Vec<std::path::PathBuf> {
    let Some(home) = dirs::home_dir() else {
        return Vec::new();
    };
    let search_dirs = [
        home.join(".OminiX").join("models"),
        home.join(".ominix").join("models"),
    ];

    let mut candidates = Vec::new();
    for dir in &search_dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            if detect_tts_variant(&path).is_some() && has_qwen3_tts_weights(&path) {
                candidates.push(path.clone());
            }

            let name = entry.file_name().to_string_lossy().to_lowercase();
            if name.contains("tts") {
                if let Ok(sub_entries) = std::fs::read_dir(&path) {
                    for sub in sub_entries.flatten() {
                        let sub_path = sub.path();
                        if sub_path.is_dir()
                            && detect_tts_variant(&sub_path).is_some()
                            && has_qwen3_tts_weights(&sub_path)
                        {
                            candidates.push(sub_path);
                        }
                    }
                }
            }
        }
    }

    candidates.sort();
    candidates.dedup();
    candidates
}

fn resolve_configured_tts_model(
    model_ref: &str,
    variant: TtsVariant,
) -> Option<std::path::PathBuf> {
    match model_config::check_model(model_ref, ModelCategory::Tts) {
        ModelAvailability::Ready {
            local_path: Some(path),
            ..
        } if path_matches_requested_model(&path, model_ref, variant) => Some(path),
        _ => None,
    }
}

fn resolve_catalog_tts_model(
    model_ref: &str,
    variant: TtsVariant,
) -> Option<std::path::PathBuf> {
    let model = crate::model_registry::get_default_models()
        .into_iter()
        .find(|model| {
            model.category == ModelCategory::Tts
                && (model.id == model_ref || model.source.repo_id.as_deref() == Some(model_ref))
        })?;
    let path = std::path::PathBuf::from(crate::utils::expand_tilde(&model.storage.local_path));
    path_matches_requested_model(&path, model_ref, variant).then_some(path)
}

fn resolve_exact_tts_model(
    model_ref: &str,
    variant: TtsVariant,
) -> Option<std::path::PathBuf> {
    let expanded = crate::utils::expand_tilde(model_ref);
    let path = std::path::PathBuf::from(&expanded);
    if path.is_dir() && path_matches_variant(&path, variant) {
        return Some(path);
    }

    if let Some(path) = resolve_configured_tts_model(model_ref, variant) {
        return Some(path);
    }

    if let Some(path) = resolve_catalog_tts_model(model_ref, variant) {
        return Some(path);
    }

    if let Some(path) = crate::utils::resolve_from_hub_cache(model_ref) {
        if path_matches_requested_model(&path, model_ref, variant) {
            return Some(path);
        }
    }

    collect_tts_candidates()
        .into_iter()
        .filter(|path| path_matches_variant(path, variant))
        .find(|path| path_matches_model_ref(path, model_ref))
}

/// Search standard model directories for a TTS model variant.
/// Searches both first-level and second-level subdirectories under each base dir.
fn find_tts_model(variant: TtsVariant) -> Option<String> {
    let mut candidates: Vec<_> = collect_tts_candidates()
        .into_iter()
        .filter(|path| path_matches_variant(path, variant))
        .collect();

    candidates.sort_by_key(|path| {
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        (!name.to_lowercase().contains("8bit"), name.len())
    });

    candidates
        .into_iter()
        .next()
        .map(|path| path.to_string_lossy().to_string())
}

// ── Engine ensure helpers ─────────────────────────────────────────

/// Ensure the engine is loaded with the requested variant. Swaps if needed.
fn ensure_variant<'a>(
    engine: &'a mut Option<qwen3_tts::Qwen3TtsEngine>,
    current_variant: &mut Option<TtsVariant>,
    current_path: &mut Option<String>,
    desired_variant: TtsVariant,
    requested_path: Option<std::path::PathBuf>,
) -> Option<&'a mut qwen3_tts::Qwen3TtsEngine> {
    let path = requested_path
        .map(|path| path.to_string_lossy().to_string())
        .or_else(|| find_tts_model(desired_variant))?;

    if *current_variant == Some(desired_variant)
        && current_path.as_deref() == Some(path.as_str())
        && engine.is_some()
    {
        return engine.as_mut();
    }

    if let Some(ref mut e) = engine {
        // Swap talker in-place (reuse decoder + tokenizer)
        tracing::info!("TTS: swapping to {desired_variant} model: {path}");
        match e.swap_model(&path) {
            Ok(()) => {
                *current_variant = Some(desired_variant);
                *current_path = Some(path);
                return engine.as_mut();
            }
            Err(e) => {
                tracing::error!("TTS: failed to swap to {desired_variant}: {e}");
                return None;
            }
        }
    }

    // Cold start — load from scratch
    tracing::info!("TTS: loading {desired_variant} model: {path}");
    match qwen3_tts::Qwen3TtsEngine::new(&path) {
        Ok(e) => {
            *engine = Some(e);
            *current_variant = Some(desired_variant);
            *current_path = Some(path);
            engine.as_mut()
        }
        Err(e) => {
            tracing::error!("TTS: failed to load {desired_variant}: {e}");
            None
        }
    }
}

/// Ensure the engine is loaded with CustomVoice variant. Swaps if needed.
fn ensure_customvoice<'a>(
    engine: &'a mut Option<qwen3_tts::Qwen3TtsEngine>,
    variant: &mut Option<TtsVariant>,
    current_path: &mut Option<String>,
) -> Option<&'a mut qwen3_tts::Qwen3TtsEngine> {
    ensure_variant(engine, variant, current_path, TtsVariant::CustomVoice, None)
}

/// Ensure the engine is loaded with Base variant (voice cloning). Swaps if needed.
fn ensure_base<'a>(
    engine: &'a mut Option<qwen3_tts::Qwen3TtsEngine>,
    variant: &mut Option<TtsVariant>,
    current_path: &mut Option<String>,
) -> Option<&'a mut qwen3_tts::Qwen3TtsEngine> {
    ensure_variant(engine, variant, current_path, TtsVariant::Base, None)
}

// ── Qwen3-TTS engine holder (used by inference thread) ─────────────

/// Holds a single Qwen3-TTS engine that swaps between CustomVoice and Base
/// variants on demand. Only one model is in memory at a time (~3 GB instead
/// of ~6 GB when both were loaded simultaneously).
pub struct Qwen3TtsEngines {
    engine: Option<qwen3_tts::Qwen3TtsEngine>,
    variant: Option<TtsVariant>,
    current_path: Option<String>,
    /// Cached reference audio samples for per-sentence voice cloning.
    cached_clone_ref: Option<Vec<f32>>,
}

impl Qwen3TtsEngines {
    /// Create and optionally eager-load the default (CustomVoice) engine.
    pub fn new(eager_load: bool) -> Self {
        let mut engines = Self {
            engine: None,
            variant: None,
            current_path: None,
            cached_clone_ref: None,
        };
        if eager_load {
            tracing::info!("Qwen3-TTS: eager-loading default model (CustomVoice)...");
            ensure_customvoice(
                &mut engines.engine,
                &mut engines.variant,
                &mut engines.current_path,
            );
            let ok = engines.engine.is_some();
            tracing::info!("Qwen3-TTS: eager load complete (loaded={ok})");
        }
        engines
    }

    /// Explicitly load the appropriate engine for a model ID.
    /// Returns Ok if the engine was loaded (or was already loaded), Err if not found.
    pub fn load_model(&mut self, model_id: &str) -> eyre::Result<String> {
        let desired_variant = TtsVariant::from_model_ref(model_id);

        // If the desired variant is already loaded, return immediately
        if self.variant == Some(desired_variant) && self.engine.is_some() {
            let path = self.current_path.clone().unwrap_or_default();
            return Ok(format!("Qwen3-TTS {desired_variant} already loaded: {path}"));
        }

        // Try exact resolution first, fall back to scanning
        let requested_path = resolve_exact_tts_model(model_id, desired_variant)
            .or_else(|| find_tts_model(desired_variant).map(std::path::PathBuf::from))
            .ok_or_else(|| {
                eyre::eyre!(
                    "Qwen3-TTS {desired_variant} model not found for '{model_id}'. Expected a model directory with qwen3_tts config and weights under ~/.OminiX/models/"
                )
            })?;
        let loaded_path = requested_path.to_string_lossy().to_string();

        match ensure_variant(
            &mut self.engine,
            &mut self.variant,
            &mut self.current_path,
            desired_variant,
            Some(requested_path),
        ) {
            Some(_) => Ok(format!("Qwen3-TTS {desired_variant} loaded: {loaded_path}")),
            None => Err(eyre::eyre!(
                "Failed to load Qwen3-TTS {desired_variant} model: {loaded_path}"
            )),
        }
    }

    pub fn unload(&mut self) -> Option<String> {
        let prev_variant = self.variant.take().map(|variant| variant.as_str().to_string());
        let prev_path = self.current_path.take();
        self.engine = None;
        self.cached_clone_ref = None;

        match (prev_variant, prev_path) {
            (Some(variant), Some(path)) => Some(format!("{variant} ({path})")),
            (Some(variant), None) => Some(variant),
            (None, Some(path)) => Some(path),
            (None, None) => None,
        }
    }

    /// Currently loaded variant name, for status reporting.
    pub fn current_variant_name(&self) -> Option<&'static str> {
        self.variant.map(TtsVariant::as_str)
    }

    /// Handle a TTS request inline (called from the inference thread).
    pub fn handle(&mut self, request: TtsRequest) {
        match request {
            TtsRequest::Speech { request, response_tx } => {
                let needs_clone = request.reference_audio.is_some();
                let result = if needs_clone {
                    (|| -> eyre::Result<Vec<u8>> {
                        let b64 = request.reference_audio.as_deref().unwrap();
                        use base64::Engine;
                        let raw = base64::engine::general_purpose::STANDARD
                            .decode(b64)
                            .map_err(|e| eyre::eyre!("Invalid base64 in reference_audio: {e}"))?;
                        let tmp = ref_audio_to_tempfile(&raw)?;
                        let engine = ensure_base(
                            &mut self.engine,
                            &mut self.variant,
                            &mut self.current_path,
                        )
                        .ok_or_else(|| eyre::eyre!("Base TTS model not found on disk"))?;
                        let lang = request.language.as_deref().unwrap_or("chinese");
                        engine.synthesize_clone(
                            &request.input,
                            tmp.path().to_str().unwrap(),
                            lang,
                            request.speed,
                            request.instruct.as_deref(),
                        )
                    })()
                } else {
                    let engine = ensure_customvoice(
                        &mut self.engine,
                        &mut self.variant,
                        &mut self.current_path,
                    );
                    match engine {
                        Some(e) => e.synthesize(&request),
                        None => Err(eyre::eyre!("CustomVoice TTS model not found on disk")),
                    }
                };
                let _ = response_tx.send(result);
            }

            TtsRequest::SpeechClone { request, response_tx } => {
                let result = (|| -> eyre::Result<Vec<u8>> {
                    let tmp = ref_audio_to_tempfile(&request.reference_audio)?;
                    let engine = ensure_base(
                        &mut self.engine,
                        &mut self.variant,
                        &mut self.current_path,
                    )
                    .ok_or_else(|| eyre::eyre!("Base TTS model not found for voice cloning"))?;
                    engine.synthesize_clone(
                        &request.input,
                        tmp.path().to_str().unwrap(),
                        &request.language,
                        request.speed,
                        request.instruct.as_deref(),
                    )
                })();
                let _ = response_tx.send(result);
            }

            TtsRequest::SpeechOneSentence { sentence, voice, language, speed, instruct, response_tx } => {
                let result = match ensure_customvoice(
                    &mut self.engine,
                    &mut self.variant,
                    &mut self.current_path,
                ) {
                    Some(engine) => engine.synthesize_one_sentence(
                        &sentence,
                        &voice,
                        &language,
                        speed,
                        instruct.as_deref(),
                    ),
                    None => Err(eyre::eyre!("CustomVoice TTS model not found")),
                };
                let _ = response_tx.send(result);
            }

            TtsRequest::PrepareCloneRef { audio_bytes, response_tx } => {
                let result = (|| -> eyre::Result<()> {
                    let tmp = ref_audio_to_tempfile(&audio_bytes)?;
                    let ref_samples = qwen3_tts::Qwen3TtsEngine::load_ref_audio(
                        tmp.path().to_str().unwrap(),
                    )?;
                    self.cached_clone_ref = Some(ref_samples);
                    // Ensure base engine is loaded
                    ensure_base(
                        &mut self.engine,
                        &mut self.variant,
                        &mut self.current_path,
                    )
                    .ok_or_else(|| eyre::eyre!("Base TTS model not found for voice cloning"))?;
                    Ok(())
                })();
                let _ = response_tx.send(result);
            }

            TtsRequest::CloneOneSentence { sentence, language, speed, instruct, response_tx } => {
                let result = if self.cached_clone_ref.is_none() {
                    Err(eyre::eyre!("No cached clone reference — call PrepareCloneRef first"))
                } else if self.engine.is_none() {
                    Err(eyre::eyre!("Base TTS model not loaded"))
                } else {
                    // Ensure Base variant is active (may need to swap back if interleaved)
                    if self.variant != Some(TtsVariant::Base) {
                        tracing::warn!("TTS: clone sentence found non-Base model loaded, swapping back");
                        if ensure_base(
                            &mut self.engine,
                            &mut self.variant,
                            &mut self.current_path,
                        )
                        .is_none()
                        {
                            let _ = response_tx.send(Err(eyre::eyre!("Failed to swap back to Base model")));
                            return;
                        }
                    }
                    let engine = self.engine.as_mut().unwrap();
                    let ref_samples = self.cached_clone_ref.as_ref().unwrap();
                    engine.synthesize_clone_one_sentence(
                        &sentence,
                        ref_samples,
                        &language,
                        speed,
                        instruct.as_deref(),
                    )
                };
                let _ = response_tx.send(result);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REF_AUDIO: &str = "/Users/yuechen/home/OminiX-MLX/step-audio2-mlx/real_speech.wav";

    /// Helper: send a TtsRequest and block on the oneshot response.
    fn send_and_recv<T: Send + 'static>(
        engines: &mut Qwen3TtsEngines,
        make_req: impl FnOnce(oneshot::Sender<eyre::Result<T>>) -> TtsRequest,
    ) -> eyre::Result<T> {
        let (tx, rx) = oneshot::channel();
        engines.handle(make_req(tx));
        rx.blocking_recv().unwrap()
    }

    /// Interleaved CustomVoice ↔ Base (xvec clone) requests.
    ///
    /// Sequence:
    ///   1. CV sentence  → loads CustomVoice
    ///   2. PrepareCloneRef → swaps to Base
    ///   3. Clone sentence #1 → stays Base
    ///   4. CV sentence (interleave!) → swaps to CustomVoice
    ///   5. Clone sentence #2 → swaps back to Base
    ///   6. CV sentence → swaps to CustomVoice
    ///
    /// Verifies: every request succeeds, variant tracking is correct,
    /// and the interleaved clone sentence recovers gracefully.
    #[test]
    #[ignore] // requires models on disk + MLX GPU
    fn should_swap_models_when_interleaved_cv_and_clone_requests() {
        // Lazy-load so we control the sequence
        let mut engines = Qwen3TtsEngines::new(false);
        assert!(engines.current_variant_name().is_none());

        // 1. CV sentence → cold-loads CustomVoice
        let pcm = send_and_recv(&mut engines, |tx| TtsRequest::SpeechOneSentence {
            sentence: "你好世界".to_string(),
            voice: "vivian".to_string(),
            language: "chinese".to_string(),
            speed: 1.0,
            instruct: None,
            response_tx: tx,
        });
        assert!(pcm.is_ok(), "CV sentence #1 failed: {:?}", pcm.err());
        assert!(!pcm.unwrap().is_empty());
        assert_eq!(engines.current_variant_name(), Some("customvoice"));

        // 2. PrepareCloneRef → swaps to Base
        let ref_bytes = std::fs::read(REF_AUDIO).expect("ref audio not found");
        let prep = send_and_recv::<()>(&mut engines, |tx| TtsRequest::PrepareCloneRef {
            audio_bytes: ref_bytes,
            response_tx: tx,
        });
        assert!(prep.is_ok(), "PrepareCloneRef failed: {:?}", prep.err());
        assert_eq!(engines.current_variant_name(), Some("base"));

        // 3. Clone sentence #1 → stays Base
        let pcm = send_and_recv(&mut engines, |tx| TtsRequest::CloneOneSentence {
            sentence: "这是克隆语音测试".to_string(),
            language: "chinese".to_string(),
            speed: 1.0,
            instruct: None,
            response_tx: tx,
        });
        assert!(pcm.is_ok(), "Clone sentence #1 failed: {:?}", pcm.err());
        assert!(!pcm.unwrap().is_empty());
        assert_eq!(engines.current_variant_name(), Some("base"));

        // 4. INTERLEAVE: CV sentence arrives mid-clone-batch → swaps to CustomVoice
        let pcm = send_and_recv(&mut engines, |tx| TtsRequest::SpeechOneSentence {
            sentence: "插入的普通语音".to_string(),
            voice: "vivian".to_string(),
            language: "chinese".to_string(),
            speed: 1.0,
            instruct: None,
            response_tx: tx,
        });
        assert!(pcm.is_ok(), "Interleaved CV sentence failed: {:?}", pcm.err());
        assert!(!pcm.unwrap().is_empty());
        assert_eq!(engines.current_variant_name(), Some("customvoice"));

        // 5. Clone sentence #2 → detects wrong variant, swaps back to Base
        let pcm = send_and_recv(&mut engines, |tx| TtsRequest::CloneOneSentence {
            sentence: "克隆语音第二句".to_string(),
            language: "chinese".to_string(),
            speed: 1.0,
            instruct: None,
            response_tx: tx,
        });
        assert!(pcm.is_ok(), "Clone sentence #2 (after interleave) failed: {:?}", pcm.err());
        assert!(!pcm.unwrap().is_empty());
        assert_eq!(engines.current_variant_name(), Some("base"));

        // 6. Back to CV
        let pcm = send_and_recv(&mut engines, |tx| TtsRequest::SpeechOneSentence {
            sentence: "最后一句普通语音".to_string(),
            voice: "vivian".to_string(),
            language: "chinese".to_string(),
            speed: 1.0,
            instruct: None,
            response_tx: tx,
        });
        assert!(pcm.is_ok(), "Final CV sentence failed: {:?}", pcm.err());
        assert!(!pcm.unwrap().is_empty());
        assert_eq!(engines.current_variant_name(), Some("customvoice"));
    }

    /// Get current process RSS in MB via macOS `task_info`.
    fn rss_mb() -> f64 {
        let output = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .expect("ps failed");
        let kb: f64 = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse()
            .unwrap_or(0.0);
        kb / 1024.0
    }

    /// Run 10 full interleave cycles (CV → Base → CV) and track RSS at each swap.
    /// Checks for memory leaks: RSS should stabilize, not grow linearly.
    #[test]
    #[ignore] // requires models on disk + MLX GPU
    fn should_not_leak_memory_across_10_interleaved_cycles() {
        let ref_bytes = std::fs::read(REF_AUDIO).expect("ref audio not found");

        let mut engines = Qwen3TtsEngines::new(false);
        let mut rss_samples: Vec<(usize, &str, f64)> = Vec::new();

        let rss_before = rss_mb();
        eprintln!("\n[mem] before load: {rss_before:.0} MB");

        for round in 0..10 {
            // --- CV sentence ---
            let pcm = send_and_recv(&mut engines, |tx| TtsRequest::SpeechOneSentence {
                sentence: format!("第{round}轮，你好世界"),
                voice: "vivian".to_string(),
                language: "chinese".to_string(),
                speed: 1.0,
                instruct: None,
                response_tx: tx,
            });
            assert!(pcm.is_ok(), "round {round} CV failed: {:?}", pcm.err());
            let rss = rss_mb();
            rss_samples.push((round, "cv", rss));
            eprintln!("[mem] round {round:>2} CV   : {rss:.0} MB");

            // --- PrepareCloneRef (swaps to Base) ---
            let prep = send_and_recv::<()>(&mut engines, |tx| TtsRequest::PrepareCloneRef {
                audio_bytes: ref_bytes.clone(),
                response_tx: tx,
            });
            assert!(prep.is_ok(), "round {round} PrepareCloneRef failed: {:?}", prep.err());

            // --- Clone sentence ---
            let pcm = send_and_recv(&mut engines, |tx| TtsRequest::CloneOneSentence {
                sentence: format!("第{round}轮，克隆语音测试"),
                language: "chinese".to_string(),
                speed: 1.0,
                instruct: None,
                response_tx: tx,
            });
            assert!(pcm.is_ok(), "round {round} clone failed: {:?}", pcm.err());
            let rss = rss_mb();
            rss_samples.push((round, "base", rss));
            eprintln!("[mem] round {round:>2} Base : {rss:.0} MB");
        }

        // Print summary
        eprintln!("\n--- RSS summary (MB) ---");
        eprintln!("round | CV     | Base");
        for round in 0..10 {
            let cv = rss_samples.iter().find(|(r, t, _)| *r == round && *t == "cv").map(|x| x.2).unwrap();
            let base = rss_samples.iter().find(|(r, t, _)| *r == round && *t == "base").map(|x| x.2).unwrap();
            eprintln!("  {round:>2}  | {cv:>6.0} | {base:>6.0}");
        }

        // Check for leaks: RSS at round 9 should not be more than 500 MB above round 1
        // (round 0 is cold-load, round 1 is the steady-state baseline)
        let baseline_cv = rss_samples.iter().find(|(r, t, _)| *r == 1 && *t == "cv").map(|x| x.2).unwrap();
        let final_cv = rss_samples.iter().find(|(r, t, _)| *r == 9 && *t == "cv").map(|x| x.2).unwrap();
        let drift = final_cv - baseline_cv;
        eprintln!("\n[mem] baseline (round 1 CV): {baseline_cv:.0} MB");
        eprintln!("[mem] final    (round 9 CV): {final_cv:.0} MB");
        eprintln!("[mem] drift: {drift:+.0} MB");

        assert!(
            drift < 500.0,
            "RSS grew by {drift:.0} MB over 10 cycles — possible memory leak"
        );
    }
}
