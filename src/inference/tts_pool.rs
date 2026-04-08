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

impl std::fmt::Display for TtsVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TtsVariant::CustomVoice => write!(f, "customvoice"),
            TtsVariant::Base => write!(f, "base"),
        }
    }
}

// ── Model discovery ────────────────────────────────────────────────

/// Search standard model directories for a TTS model variant.
/// Searches both first-level and second-level subdirectories under each base dir.
fn find_tts_model(variant: &str) -> Option<String> {
    fn has_qwen3_tts_weights(path: &std::path::Path) -> bool {
        path.join("model.safetensors").is_file()
            || path.join("model.safetensors.index.json").is_file()
    }

    let home = dirs::home_dir()?;
    let search_dirs = [
        home.join(".OminiX").join("models"),
        home.join(".ominix").join("models"),
    ];
    for dir in &search_dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            let mut candidates = Vec::new();
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_lowercase();
                if !path.is_dir() {
                    continue;
                }
                // Direct match (first level)
                if name.contains("tts") && name.contains(variant) {
                    candidates.push(path.clone());
                }
                // Search one level deeper (e.g. ~/.OminiX/models/qwen3-tts-mlx/<variant-dir>)
                if name.contains("tts") {
                    if let Ok(sub_entries) = std::fs::read_dir(&path) {
                        for sub in sub_entries.flatten() {
                            let sub_path = sub.path();
                            let sub_name = sub.file_name().to_string_lossy().to_lowercase();
                            if sub_path.is_dir()
                                && sub_name.contains(variant)
                                && has_qwen3_tts_weights(&sub_path)
                            {
                                candidates.push(sub_path);
                            }
                        }
                    }
                }
            }

            candidates.sort_by_key(|path| {
                let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                (
                    !has_qwen3_tts_weights(path),
                    !name.to_lowercase().contains("8bit"),
                    name.len(),
                )
            });

            if let Some(path) = candidates.into_iter().next() {
                return Some(path.to_string_lossy().to_string());
            }
        }
    }
    None
}

// ── Engine ensure helpers ─────────────────────────────────────────

/// Ensure the engine is loaded with CustomVoice variant. Swaps if needed.
fn ensure_customvoice<'a>(
    engine: &'a mut Option<qwen3_tts::Qwen3TtsEngine>,
    variant: &mut Option<TtsVariant>,
) -> Option<&'a mut qwen3_tts::Qwen3TtsEngine> {
    if *variant == Some(TtsVariant::CustomVoice) && engine.is_some() {
        return engine.as_mut();
    }

    let path = find_tts_model("customvoice")?;

    if let Some(ref mut e) = engine {
        // Swap talker in-place (reuse decoder + tokenizer)
        tracing::info!("TTS: swapping to CustomVoice model: {path}");
        match e.swap_model(&path) {
            Ok(()) => {
                *variant = Some(TtsVariant::CustomVoice);
                return engine.as_mut();
            }
            Err(e) => {
                tracing::error!("TTS: failed to swap to CustomVoice: {e}");
                return None;
            }
        }
    }

    // Cold start — load from scratch
    tracing::info!("TTS: loading CustomVoice model: {path}");
    match qwen3_tts::Qwen3TtsEngine::new(&path) {
        Ok(e) => {
            *engine = Some(e);
            *variant = Some(TtsVariant::CustomVoice);
            engine.as_mut()
        }
        Err(e) => {
            tracing::error!("TTS: failed to load CustomVoice: {e}");
            None
        }
    }
}

/// Ensure the engine is loaded with Base variant (voice cloning). Swaps if needed.
fn ensure_base<'a>(
    engine: &'a mut Option<qwen3_tts::Qwen3TtsEngine>,
    variant: &mut Option<TtsVariant>,
) -> Option<&'a mut qwen3_tts::Qwen3TtsEngine> {
    if *variant == Some(TtsVariant::Base) && engine.is_some() {
        return engine.as_mut();
    }

    let path = find_tts_model("base")?;

    if let Some(ref mut e) = engine {
        // Swap talker in-place (reuse decoder + tokenizer)
        tracing::info!("TTS: swapping to Base model: {path}");
        match e.swap_model(&path) {
            Ok(()) => {
                *variant = Some(TtsVariant::Base);
                return engine.as_mut();
            }
            Err(e) => {
                tracing::error!("TTS: failed to swap to Base: {e}");
                return None;
            }
        }
    }

    // Cold start — load from scratch
    tracing::info!("TTS: loading Base model: {path}");
    match qwen3_tts::Qwen3TtsEngine::new(&path) {
        Ok(e) => {
            *engine = Some(e);
            *variant = Some(TtsVariant::Base);
            engine.as_mut()
        }
        Err(e) => {
            tracing::error!("TTS: failed to load Base model: {e}");
            None
        }
    }
}

// ── Qwen3-TTS engine holder (used by inference thread) ─────────────

/// Holds a single Qwen3-TTS engine that swaps between CustomVoice and Base
/// variants on demand. Only one model is in memory at a time (~3 GB instead
/// of ~6 GB when both were loaded simultaneously).
pub struct Qwen3TtsEngines {
    engine: Option<qwen3_tts::Qwen3TtsEngine>,
    variant: Option<TtsVariant>,
    /// Cached reference audio samples for per-sentence voice cloning.
    cached_clone_ref: Option<Vec<f32>>,
}

impl Qwen3TtsEngines {
    /// Create and optionally eager-load the default (CustomVoice) engine.
    pub fn new(eager_load: bool) -> Self {
        let mut engines = Self {
            engine: None,
            variant: None,
            cached_clone_ref: None,
        };
        if eager_load {
            tracing::info!("Qwen3-TTS: eager-loading default model (CustomVoice)...");
            ensure_customvoice(&mut engines.engine, &mut engines.variant);
            let ok = engines.engine.is_some();
            tracing::info!("Qwen3-TTS: eager load complete (loaded={ok})");
        }
        engines
    }

    /// Explicitly load the appropriate engine for a model ID.
    /// Returns Ok if the engine was loaded (or was already loaded), Err if not found.
    pub fn load_model(&mut self, model_id: &str) -> eyre::Result<String> {
        if model_id.contains("base") {
            match ensure_base(&mut self.engine, &mut self.variant) {
                Some(_) => Ok("Qwen3-TTS Base loaded".to_string()),
                None => Err(eyre::eyre!("Qwen3-TTS Base model not found on disk. Expected a directory containing 'tts' and 'base' under ~/.OminiX/models/")),
            }
        } else {
            match ensure_customvoice(&mut self.engine, &mut self.variant) {
                Some(_) => Ok("Qwen3-TTS CustomVoice loaded".to_string()),
                None => Err(eyre::eyre!("Qwen3-TTS CustomVoice model not found on disk. Expected a directory containing 'tts' and 'customvoice' under ~/.OminiX/models/")),
            }
        }
    }

    /// Currently loaded variant name, for status reporting.
    pub fn current_variant_name(&self) -> Option<&'static str> {
        self.variant.map(|v| match v {
            TtsVariant::CustomVoice => "customvoice",
            TtsVariant::Base => "base",
        })
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
                        let engine = ensure_base(&mut self.engine, &mut self.variant)
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
                    let engine = ensure_customvoice(&mut self.engine, &mut self.variant);
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
                    let engine = ensure_base(&mut self.engine, &mut self.variant)
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
                let result = match ensure_customvoice(&mut self.engine, &mut self.variant) {
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
                    ensure_base(&mut self.engine, &mut self.variant)
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
                        if ensure_base(&mut self.engine, &mut self.variant).is_none() {
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
