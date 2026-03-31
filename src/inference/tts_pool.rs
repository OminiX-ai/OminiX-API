//! Qwen3-TTS engine management and request types.
//!
//! TTS requests are handled inline by the inference thread (not a separate pool)
//! to serialize all GPU access through a single thread. This prevents Metal
//! command buffer crashes when ASR and TTS run concurrently on separate threads.
//!
//! See: <https://github.com/ml-explore/mlx/issues/3078> (MLX thread safety)

use tokio::sync::{mpsc, oneshot};

use crate::engines::qwen3_tts;
use crate::types::{SpeechCloneRequest, SpeechRequest};

use super::AudioChunk;

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
    /// Streaming speech — yields PCM chunks via channel.
    SpeechStream {
        request: SpeechRequest,
        chunk_tx: mpsc::Sender<AudioChunk>,
    },
    /// Voice cloning (returns complete WAV).
    SpeechClone {
        request: SpeechCloneRequest,
        response_tx: oneshot::Sender<eyre::Result<Vec<u8>>>,
    },
    /// Streaming voice cloning — yields PCM chunks per sentence.
    SpeechCloneStream {
        request: SpeechCloneRequest,
        chunk_tx: mpsc::Sender<AudioChunk>,
    },
}

/// Configuration for the TTS worker pool.
#[derive(Debug, Clone)]
pub struct TtsPoolConfig {
    /// When true (default), load both CustomVoice and Base models at startup
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

// ── Model discovery ────────────────────────────────────────────────

/// Search standard model directories for a TTS model variant.
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
                if path.is_dir() && name.contains("tts") && name.contains(variant) {
                    candidates.push(path);
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


/// Ensure CustomVoice engine is loaded, return mutable ref.
fn ensure_customvoice(
    engine: &mut Option<qwen3_tts::Qwen3TtsEngine>,
    worker_id: usize,
) -> Option<&mut qwen3_tts::Qwen3TtsEngine> {
    if engine.as_ref().is_some_and(|e| e.supports_preset_speakers()) {
        return engine.as_mut();
    }
    // Need to load or switch
    if let Some(path) = find_tts_model("customvoice") {
        tracing::info!("TTS worker {worker_id}: loading CustomVoice model: {path}");
        match qwen3_tts::Qwen3TtsEngine::new(&path) {
            Ok(e) => {
                *engine = Some(e);
                engine.as_mut()
            }
            Err(e) => {
                tracing::error!("TTS worker {worker_id}: failed to load CustomVoice: {e}");
                None
            }
        }
    } else {
        // Fall back to whatever is loaded
        engine.as_mut()
    }
}

/// Ensure Base engine (voice cloning) is loaded, return mutable ref.
fn ensure_base(
    engine: &mut Option<qwen3_tts::Qwen3TtsEngine>,
    worker_id: usize,
) -> Option<&mut qwen3_tts::Qwen3TtsEngine> {
    if engine
        .as_ref()
        .is_some_and(|e| e.supports_voice_cloning())
    {
        return engine.as_mut();
    }
    if let Some(path) = find_tts_model("base") {
        tracing::info!("TTS worker {worker_id}: loading Base model: {path}");
        match qwen3_tts::Qwen3TtsEngine::new(&path) {
            Ok(e) => {
                *engine = Some(e);
                engine.as_mut()
            }
            Err(e) => {
                tracing::error!("TTS worker {worker_id}: failed to load Base model: {e}");
                None
            }
        }
    } else {
        None
    }
}

// ── Qwen3-TTS engine holder (used by inference thread) ─────────────

/// Holds both Qwen3-TTS engine variants for use by the inference thread.
pub struct Qwen3TtsEngines {
    cv_engine: Option<qwen3_tts::Qwen3TtsEngine>,
    base_engine: Option<qwen3_tts::Qwen3TtsEngine>,
}

impl Qwen3TtsEngines {
    /// Create and optionally eager-load both engine variants.
    pub fn new(eager_load: bool) -> Self {
        let mut engines = Self {
            cv_engine: None,
            base_engine: None,
        };
        if eager_load {
            tracing::info!("Qwen3-TTS: eager-loading models...");
            ensure_customvoice(&mut engines.cv_engine, 0);
            ensure_base(&mut engines.base_engine, 0);
            let cv_ok = engines.cv_engine.is_some();
            let base_ok = engines.base_engine.is_some();
            tracing::info!("Qwen3-TTS: eager load complete (CustomVoice={cv_ok}, Base={base_ok})");
        }
        engines
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
                        let engine = ensure_base(&mut self.base_engine, 0)
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
                    let engine = ensure_customvoice(&mut self.cv_engine, 0);
                    match engine {
                        Some(e) => e.synthesize(&request),
                        None => Err(eyre::eyre!("CustomVoice TTS model not found on disk")),
                    }
                };
                let _ = response_tx.send(result);
            }

            TtsRequest::SpeechStream { request, chunk_tx } => {
                let engine = ensure_customvoice(&mut self.cv_engine, 0);
                if let Some(engine) = engine {
                    let tx = chunk_tx.clone();
                    let result = engine.synthesize_sentences(&request, |pcm_bytes| {
                        tx.blocking_send(AudioChunk::Pcm(pcm_bytes.to_vec())).is_ok()
                    });
                    match result {
                        Ok(()) => {
                            let _ = chunk_tx.blocking_send(AudioChunk::Done {
                                total_samples: 0,
                                duration_secs: 0.0,
                            });
                        }
                        Err(e) => {
                            let _ = chunk_tx.blocking_send(AudioChunk::Error(e.to_string()));
                        }
                    }
                } else {
                    let _ = chunk_tx
                        .blocking_send(AudioChunk::Error("TTS model not loaded".to_string()));
                }
            }

            TtsRequest::SpeechClone { request, response_tx } => {
                let result = (|| -> eyre::Result<Vec<u8>> {
                    let tmp = ref_audio_to_tempfile(&request.reference_audio)?;
                    let engine = ensure_base(&mut self.base_engine, 0)
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

            TtsRequest::SpeechCloneStream { request, chunk_tx } => {
                let tmp = match ref_audio_to_tempfile(&request.reference_audio) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = chunk_tx.blocking_send(AudioChunk::Error(e.to_string()));
                        return;
                    }
                };
                let engine = ensure_base(&mut self.base_engine, 0);
                if let Some(engine) = engine {
                    let tx = chunk_tx.clone();
                    let result = engine.synthesize_clone_sentences(
                        &request.input,
                        tmp.path().to_str().unwrap(),
                        &request.language,
                        request.speed,
                        request.instruct.as_deref(),
                        |pcm_bytes| tx.blocking_send(AudioChunk::Pcm(pcm_bytes.to_vec())).is_ok(),
                    );
                    match result {
                        Ok(()) => {
                            let _ = chunk_tx.blocking_send(AudioChunk::Done {
                                total_samples: 0,
                                duration_secs: 0.0,
                            });
                        }
                        Err(e) => {
                            let _ = chunk_tx.blocking_send(AudioChunk::Error(e.to_string()));
                        }
                    }
                } else {
                    let _ = chunk_tx.blocking_send(AudioChunk::Error(
                        "Base TTS model not found for voice cloning".to_string(),
                    ));
                }
            }
        }
    }
}
