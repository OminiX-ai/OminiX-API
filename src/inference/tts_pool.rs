//! TTS worker pool.
//!
//! Runs TTS inference on a dedicated thread to avoid blocking the main
//! inference thread (LLM/ASR/image).
//!
//! ## Why single-worker only (no dynamic scaling)
//!
//! Metal's `MTLCommandQueue` is thread-safe and supports encoding command
//! buffers from multiple CPU threads. However, **MLX does not support
//! concurrent inference from separate threads** — its Metal backend has
//! thread-safety issues where multiple threads race on shared
//! `DeviceStream::buffer` and `DeviceStream::encoder` fields, causing
//! crashes or undefined behavior.
//!
//! See:
//! - <https://github.com/ml-explore/mlx/issues/3078> (concurrent inference)
//! - <https://github.com/ml-explore/mlx/issues/2133> (thread safety tracking)
//! - <https://github.com/ml-explore/mlx/pull/2104>  (partial Metal fix)
//!
//! Once MLX adds proper multi-stream thread safety, the scaling code below
//! (currently commented out) can be re-enabled to spawn additional workers.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::{mpsc, oneshot};

use crate::engines::qwen3_tts;
use crate::types::{SpeechCloneRequest, SpeechRequest};

use super::AudioChunk;

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
    // NOTE: Scaling fields kept for future use when MLX supports concurrent
    // inference from multiple threads. Currently only 1 worker is used.
    //
    // /// Maximum number of concurrent TTS worker threads.
    // pub max_workers: usize,
    // /// Seconds of idle time before a surplus worker is reclaimed.
    // pub idle_timeout_secs: u64,
    // /// Minimum free RAM (GB) required before spawning a new worker.
    // pub min_free_ram_gb: f64,
    // /// Approximate RAM per TTS engine instance (GB).
    // pub ram_per_instance_gb: f64,
}

impl Default for TtsPoolConfig {
    fn default() -> Self {
        Self {}
    }
}

impl TtsPoolConfig {
    pub fn from_env() -> Self {
        Self::default()
    }
}

// ── Pool internals ──────────────────────────────────────────────────

struct WorkerHandle {
    tx: std::sync::mpsc::Sender<TtsRequest>,
    /// Number of requests queued + in-flight for this worker.
    /// Incremented by pool manager on dispatch, decremented by worker on completion.
    queued: Arc<AtomicU64>,
    /// Epoch seconds of last completed request.
    last_active: Arc<AtomicU64>,
    #[allow(dead_code)]
    thread: Option<std::thread::JoinHandle<()>>,
}

/// Search standard model directories for a TTS model variant.
fn find_tts_model(variant: &str) -> Option<String> {
    let home = dirs::home_dir()?;
    let search_dirs = [
        home.join(".OminiX").join("models"),
        home.join(".ominix").join("models"),
    ];
    for dir in &search_dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy().to_lowercase();
                if entry.path().is_dir() && name_str.contains("tts") && name_str.contains(variant)
                {
                    return Some(entry.path().to_string_lossy().to_string());
                }
            }
        }
    }
    None
}

fn now_epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// NOTE: available_ram_gb() and parse_vm_stat_value() removed — they were
// only used by the multi-worker scaling logic. Re-add from git history
// when MLX supports concurrent inference.

// ── Worker thread ───────────────────────────────────────────────────

fn spawn_worker(id: usize) -> WorkerHandle {
    let queued = Arc::new(AtomicU64::new(0));
    let last_active = Arc::new(AtomicU64::new(now_epoch_secs()));

    let (tx, rx) = std::sync::mpsc::channel::<TtsRequest>();

    let queued_clone = queued.clone();
    let last_active_clone = last_active.clone();

    let thread = std::thread::Builder::new()
        .name(format!("tts-worker-{id}"))
        .spawn(move || {
            worker_main(id, rx, queued_clone, last_active_clone);
        })
        .expect("failed to spawn TTS worker thread");

    WorkerHandle {
        tx,
        queued,
        last_active,
        thread: Some(thread),
    }
}

fn worker_main(
    id: usize,
    rx: std::sync::mpsc::Receiver<TtsRequest>,
    queued: Arc<AtomicU64>,
    last_active: Arc<AtomicU64>,
) {
    tracing::info!("TTS worker {id} started");

    // Lazy-loaded engine slots (one per model variant).
    let mut cv_engine: Option<qwen3_tts::Qwen3TtsEngine> = None;
    let mut base_engine: Option<qwen3_tts::Qwen3TtsEngine> = None;

    while let Ok(request) = rx.recv() {

        match request {
            TtsRequest::Speech {
                request,
                response_tx,
            } => {
                let needs_clone = request.reference_audio.is_some();
                let result = if needs_clone {
                    let engine = ensure_base(&mut base_engine, id);
                    match engine {
                        Some(e) => {
                            let ref_audio = request.reference_audio.as_deref().unwrap();
                            let lang = request.language.as_deref().unwrap_or("chinese");
                            e.synthesize_clone(&request.input, ref_audio, lang, request.speed)
                        }
                        None => Err(eyre::eyre!("Base TTS model not found on disk")),
                    }
                } else {
                    let engine = ensure_customvoice(&mut cv_engine, id);
                    match engine {
                        Some(e) => e.synthesize(&request),
                        None => Err(eyre::eyre!("CustomVoice TTS model not found on disk")),
                    }
                };
                let _ = response_tx.send(result);
            }

            TtsRequest::SpeechStream { request, chunk_tx } => {
                // Sentence-level streaming: split text into sentences, synthesize
                // each independently (full decode per sentence = no artifacts),
                // and send PCM as each sentence completes. Client receives first
                // audio after ~2-3s instead of waiting for full synthesis.
                let engine = ensure_customvoice(&mut cv_engine, id);
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

            TtsRequest::SpeechClone {
                request,
                response_tx,
            } => {
                let engine = ensure_base(&mut base_engine, id);
                let result = match engine {
                    Some(e) => e.synthesize_clone(
                        &request.input,
                        &request.reference_audio,
                        &request.language,
                        request.speed,
                    ),
                    None => Err(eyre::eyre!("Base TTS model not found for voice cloning")),
                };
                let _ = response_tx.send(result);
            }

            TtsRequest::SpeechCloneStream {
                request,
                chunk_tx,
            } => {
                let engine = ensure_base(&mut base_engine, id);
                if let Some(engine) = engine {
                    let tx = chunk_tx.clone();
                    let result = engine.synthesize_clone_sentences(
                        &request.input,
                        &request.reference_audio,
                        &request.language,
                        request.speed,
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

        queued.fetch_sub(1, Ordering::Release);
        last_active.store(now_epoch_secs(), Ordering::Release);
    }

    tracing::info!("TTS worker {id} shutting down");
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

// ── Pool manager ────────────────────────────────────────────────────

/// Run the TTS pool manager on a dedicated thread.
/// Receives `TtsRequest`s and dispatches to a single worker thread.
///
/// Only one worker is used because MLX lacks thread-safe concurrent
/// inference (see module docs). All requests queue to worker 0.
pub fn run_pool(mut rx: mpsc::Receiver<TtsRequest>, _config: TtsPoolConfig) {
    tracing::info!("TTS pool started (single worker — MLX is not thread-safe for concurrent inference)");

    let worker = spawn_worker(0);

    while let Some(request) = rx.blocking_recv() {
        worker.queued.fetch_add(1, Ordering::Release);
        if worker.tx.send(request).is_err() {
            tracing::error!("TTS worker 0 died — no recovery, dropping request");
        }
    }

    tracing::info!("TTS pool shutting down");
    drop(worker);
}

// ── Commented-out scaling code ─────────────────────────────────────
// Re-enable when MLX supports concurrent inference from multiple
// threads (see https://github.com/ml-explore/mlx/issues/3078).
//
// fn can_spawn_worker(config: &TtsPoolConfig) -> bool {
//     let free = available_ram_gb();
//     let needed = config.min_free_ram_gb + config.ram_per_instance_gb;
//     if free < needed {
//         tracing::info!(
//             "TTS pool: not enough RAM to spawn worker (free: {free:.1}GB, need: {needed:.1}GB)"
//         );
//         return false;
//     }
//     true
// }
//
// fn reap_idle_workers(workers: &mut Vec<WorkerHandle>, config: &TtsPoolConfig) {
//     if workers.len() <= 1 {
//         return;
//     }
//     let now = now_epoch_secs();
//     let timeout = config.idle_timeout_secs;
//     let mut i = workers.len();
//     while i > 1 {
//         i -= 1;
//         let w = &workers[i];
//         if w.queued.load(Ordering::Acquire) == 0 {
//             let idle_secs = now.saturating_sub(w.last_active.load(Ordering::Acquire));
//             if idle_secs >= timeout {
//                 tracing::info!(
//                     "TTS pool: reaping idle worker {i} (idle {idle_secs}s, pool: {} -> {})",
//                     workers.len(),
//                     workers.len() - 1
//                 );
//                 workers.remove(i);
//             }
//         }
//     }
// }
