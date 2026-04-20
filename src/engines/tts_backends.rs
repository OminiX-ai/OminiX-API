//! Concrete `TextToSpeech` implementations for the four v1 backends.
//!
//! Contract: `ASCEND_API_BRIDGE_CONTRACT.md` §5 B3.2.
//!
//! ## Backend index
//!
//! | Impl                   | Transport                  | Platform          |
//! |------------------------|----------------------------|-------------------|
//! | `AscendSubprocessTts`  | `qwen_tts` binary (Popen)  | any (fork/exec)   |
//! | `AscendFfiTts`         | `libqwen_tts_api.so` FFI   | Linux + feature   |
//! | `Qwen3MlxTts`          | inference-thread channel   | Mac/Apple Silicon |
//! | `GptSovitsMlxTts`      | inference-thread channel   | Mac/Apple Silicon |
//!
//! The MLX impls cheat slightly: they ship a `tokio::sync::mpsc::Sender`
//! handle for the inference thread, use `tokio::runtime::Handle::current()
//! .block_on(...)` to wait on the `oneshot` reply, and expose a synchronous
//! `TextToSpeech::synthesize` per the contract signature. Callers are
//! expected to invoke from `tokio::task::spawn_blocking` (which all four
//! existing Ascend handlers already do), so the `block_on` is safe.
//!
//! ## Platform gating rationale
//!
//! `AscendFfiTts` requires `qwen_tts_ascend_sys` to be link-clean against
//! `libqwen_tts_api.so`, which only ships for Linux/aarch64. The trait
//! *selector* (see `build_tts_backend`) must still compile on Mac so
//! `cargo check` on dev hosts works — we provide a stub that returns
//! `Unsupported` plus a startup warning when FFI is requested on Mac.

// Same dead-code rationale as `tts_trait.rs`: with `ascend_config` wired
// to `None` in `main.rs`, the trait impls live here but aren't reached
// from the binary on Mac dev builds. Keep them exported so the Ascend
// host (and the B4 integration test) can pick them up without touching
// these files.
#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use tokio::sync::{mpsc, oneshot};

use crate::engines::ascend::{AscendConfig, AscendTtsEngine};
use crate::engines::tts_trait::{TextToSpeech, TtsCloneRequest, TtsError, TtsRequest, TtsResponse};
use crate::inference::tts_pool::TtsRequest as InferenceTtsRequest;
use crate::inference::InferenceRequest;
use crate::types::{SpeechCloneRequest, SpeechRequest};

// ============================================================================
// 1. Ascend subprocess — refactor of existing AscendTtsEngine
// ============================================================================

/// Wraps the existing subprocess-based `AscendTtsEngine`. The underlying
/// engine is unchanged; we only move its call-site behind the trait.
/// Preserves byte-identical output from `qwen_tts` per contract §5.3.4.
pub struct AscendSubprocessTts {
    /// Shared config — cloned per-request since the current engine
    /// re-validates paths in `AscendTtsEngine::new`. Keeping the Arc lets
    /// us avoid deep-cloning the whole config on each call.
    config: Arc<AscendConfig>,
}

impl AscendSubprocessTts {
    pub fn new(config: Arc<AscendConfig>) -> Self {
        Self { config }
    }

    fn build_engine(&self) -> Result<AscendTtsEngine, TtsError> {
        AscendTtsEngine::new((*self.config).clone()).map_err(TtsError::from)
    }
}

impl TextToSpeech for AscendSubprocessTts {
    fn backend_name(&self) -> &'static str {
        "ascend-subprocess"
    }

    fn supports_clone(&self) -> bool {
        true
    }

    fn synthesize(&self, req: TtsRequest) -> Result<TtsResponse, TtsError> {
        // Adapt to the existing `&SpeechRequest`-shaped argument so the
        // underlying engine code path is untouched.
        let speech_req = SpeechRequest {
            model: None,
            input: req.input,
            voice: Some(req.voice),
            response_format: "wav".to_string(),
            speed: req.speed,
            reference_audio: None,
            language: Some(req.language),
            instruct: req.instruct,
        };
        let engine = self.build_engine()?;
        let wav = engine.synthesize(&speech_req).map_err(TtsError::from)?;
        Ok(TtsResponse::Wav(wav))
    }

    fn synthesize_clone(&self, req: TtsCloneRequest) -> Result<TtsResponse, TtsError> {
        let engine = self.build_engine()?;
        let wav = engine
            .synthesize_clone(
                &req.input,
                &req.reference_audio,
                &req.language,
                req.speed,
                req.instruct.as_deref(),
            )
            .map_err(TtsError::from)?;
        Ok(TtsResponse::Wav(wav))
    }
}

// ============================================================================
// 2. Ascend FFI — wraps qwen_tts_ascend_sys::QwenTtsCtx
// ============================================================================

/// Real FFI impl behind the `ascend-tts-ffi` feature on Linux targets.
///
/// The `QwenTtsCtx` handle is `!Sync` (see crate-level docs in
/// `qwen-tts-ascend-sys`): the engine's KV cache mutates on every
/// `forward`/`predict_codes` call, so two threads touching the same
/// handle corrupt it. We serialize with a `Mutex` inside the wrapper so
/// the trait's `&self` contract holds (risk register §7).
///
/// **B3 scope note:** this impl currently returns `Unsupported` because
/// end-to-end synthesis via the FFI ABI (forward loop + codec predict +
/// decode audio) requires wiring that belongs in B4/C1 — it is not a
/// drop-in replacement for the subprocess binary yet. The struct, the
/// lock shape, and the trait binding all land now so the handler
/// refactor (§5.3.3) has a concrete type to dispatch to; the inner
/// generation loop will fill in on the Ascend host where we can
/// validate against real weights.
#[cfg(all(feature = "ascend-tts-ffi", target_os = "linux"))]
pub struct AscendFfiTts {
    /// Path to the TTS model directory. Stored so `synthesize` can lazy-
    /// load if/when the inner loop is wired up.
    #[allow(dead_code)]
    model_dir: std::path::PathBuf,
    /// Serialization lock around the `!Sync` context handle. `Option`
    /// because we defer the actual `qwen_tts_load` to the first call
    /// (cold-start cost matches the subprocess path).
    ctx: Mutex<Option<qwen_tts_ascend_sys::QwenTtsCtx>>,
    /// GPU layers and threads from `AscendConfig`, passed to
    /// `qwen_tts_load` on first use.
    n_gpu_layers: i32,
    n_threads: i32,
}

#[cfg(all(feature = "ascend-tts-ffi", target_os = "linux"))]
impl AscendFfiTts {
    pub fn new(config: Arc<AscendConfig>) -> Result<Self, TtsError> {
        let model_dir = config.tts_model_dir.as_ref().ok_or_else(|| {
            TtsError::Backend("ASCEND_TTS_MODEL_DIR not set".to_string())
        })?;
        Ok(Self {
            model_dir: model_dir.clone(),
            ctx: Mutex::new(None),
            n_gpu_layers: config.gpu_layers as i32,
            n_threads: config.threads as i32,
        })
    }
}

#[cfg(all(feature = "ascend-tts-ffi", target_os = "linux"))]
impl TextToSpeech for AscendFfiTts {
    fn backend_name(&self) -> &'static str {
        "ascend-ffi"
    }

    fn supports_clone(&self) -> bool {
        // Clone will be supported once the generation loop lands; gate
        // on has_speaker_encoder at call time.
        false
    }

    fn synthesize(&self, _req: TtsRequest) -> Result<TtsResponse, TtsError> {
        // B3 exposes the dispatch shape; the inner generation loop is a
        // B4 follow-up (needs real-hardware validation).
        let _ = (&self.ctx, self.n_gpu_layers, self.n_threads);
        Err(TtsError::Unsupported(
            "AscendFfiTts::synthesize: generation loop not wired (B4 follow-up)",
        ))
    }
}

/// Stub for non-Linux targets so the enum-selector code compiles on Mac.
/// See `build_tts_backend` for how this is selected / fallback-warned.
#[cfg(not(all(feature = "ascend-tts-ffi", target_os = "linux")))]
pub struct AscendFfiTts {
    _never: std::marker::PhantomData<*const ()>,
}

#[cfg(not(all(feature = "ascend-tts-ffi", target_os = "linux")))]
impl AscendFfiTts {
    #[allow(dead_code)]
    pub fn new(_config: Arc<AscendConfig>) -> Result<Self, TtsError> {
        Err(TtsError::Unsupported(
            "AscendFfiTts not available: build with --features ascend-tts-ffi on linux",
        ))
    }
}

// SAFETY: the stub is a ZST-ish marker; never instantiated. Needed so
// `Arc<dyn TextToSpeech>` has the same type erasure shape across platforms.
#[cfg(not(all(feature = "ascend-tts-ffi", target_os = "linux")))]
unsafe impl Send for AscendFfiTts {}
#[cfg(not(all(feature = "ascend-tts-ffi", target_os = "linux")))]
unsafe impl Sync for AscendFfiTts {}

#[cfg(not(all(feature = "ascend-tts-ffi", target_os = "linux")))]
impl TextToSpeech for AscendFfiTts {
    fn backend_name(&self) -> &'static str {
        "ascend-ffi-stub"
    }
    fn supports_clone(&self) -> bool {
        false
    }
    fn synthesize(&self, _req: TtsRequest) -> Result<TtsResponse, TtsError> {
        Err(TtsError::Unsupported(
            "AscendFfiTts stub: feature=ascend-tts-ffi + target_os=linux required",
        ))
    }
}

// ============================================================================
// 3. Qwen3-TTS MLX — channel-backed
// ============================================================================

/// Wraps the existing Qwen3-TTS MLX inference-thread path. Because the
/// MLX engine is `!Send` (Metal context), the real work happens on the
/// dedicated inference thread; we only hold a `Sender` and do a
/// `block_on` on the `oneshot` reply.
///
/// Callers must invoke from `spawn_blocking` context (existing handlers
/// already do) so the `block_on` doesn't deadlock the tokio runtime.
pub struct Qwen3MlxTts {
    inference_tx: mpsc::Sender<InferenceRequest>,
}

impl Qwen3MlxTts {
    pub fn new(inference_tx: mpsc::Sender<InferenceRequest>) -> Self {
        Self { inference_tx }
    }

    /// Shared helper: send an InferenceRequest::Qwen3Tts, block on reply.
    fn dispatch(
        &self,
        build: impl FnOnce(oneshot::Sender<eyre::Result<Vec<u8>>>) -> InferenceTtsRequest,
    ) -> Result<TtsResponse, TtsError> {
        let (tx, rx) = oneshot::channel();
        let req = build(tx);
        self.inference_tx
            .blocking_send(InferenceRequest::Qwen3Tts(req))
            .map_err(|e| TtsError::Backend(format!("inference channel closed: {e}")))?;
        let result = rx
            .blocking_recv()
            .map_err(|e| TtsError::Backend(format!("inference dropped reply: {e}")))?;
        let wav = result.map_err(TtsError::from)?;
        Ok(TtsResponse::Wav(wav))
    }
}

impl TextToSpeech for Qwen3MlxTts {
    fn backend_name(&self) -> &'static str {
        "qwen3-mlx"
    }

    fn supports_clone(&self) -> bool {
        true
    }

    fn synthesize(&self, req: TtsRequest) -> Result<TtsResponse, TtsError> {
        // Map through SpeechRequest — the existing TtsRequest::Speech arm
        // reads this shape. `response_format=wav` matches what the MLX
        // engine emits (hound-written WAV) and what handlers expect.
        let speech_req = SpeechRequest {
            model: None,
            input: req.input,
            voice: Some(req.voice),
            response_format: "wav".to_string(),
            speed: req.speed,
            reference_audio: None,
            language: Some(req.language),
            instruct: req.instruct,
        };
        self.dispatch(|response_tx| InferenceTtsRequest::Speech {
            request: speech_req,
            response_tx,
        })
    }

    fn synthesize_clone(&self, req: TtsCloneRequest) -> Result<TtsResponse, TtsError> {
        let clone_req = SpeechCloneRequest {
            input: req.input,
            reference_audio: req.reference_audio,
            language: req.language,
            speed: req.speed,
            instruct: req.instruct,
        };
        self.dispatch(|response_tx| InferenceTtsRequest::SpeechClone {
            request: clone_req,
            response_tx,
        })
    }
}

// ============================================================================
// 4. GPT-SoVITS MLX — channel-backed via InferenceRequest::Speech
// ============================================================================

/// Wraps the legacy GPT-SoVITS MLX path, which flows through
/// `InferenceRequest::Speech` (not `Qwen3Tts`). Same block_on pattern as
/// `Qwen3MlxTts`.
///
/// Kept deliberately thin: the existing TTS pool / engine routing lives
/// in `inference::tts_pool` and we don't want to fork that logic here.
pub struct GptSovitsMlxTts {
    inference_tx: mpsc::Sender<InferenceRequest>,
}

impl GptSovitsMlxTts {
    pub fn new(inference_tx: mpsc::Sender<InferenceRequest>) -> Self {
        Self { inference_tx }
    }
}

impl TextToSpeech for GptSovitsMlxTts {
    fn backend_name(&self) -> &'static str {
        "gpt-sovits-mlx"
    }

    fn supports_clone(&self) -> bool {
        // GPT-SoVITS always clones from a reference; `synthesize` here
        // uses a preset voice pickle, `synthesize_clone` uses raw bytes.
        true
    }

    fn synthesize(&self, req: TtsRequest) -> Result<TtsResponse, TtsError> {
        let speech_req = SpeechRequest {
            model: None,
            input: req.input,
            voice: Some(req.voice),
            response_format: "wav".to_string(),
            speed: req.speed,
            reference_audio: None,
            language: Some(req.language),
            instruct: req.instruct,
        };
        let (tx, rx) = oneshot::channel();
        self.inference_tx
            .blocking_send(InferenceRequest::Speech {
                request: speech_req,
                response_tx: tx,
            })
            .map_err(|e| TtsError::Backend(format!("inference channel closed: {e}")))?;
        let wav = rx
            .blocking_recv()
            .map_err(|e| TtsError::Backend(format!("inference dropped reply: {e}")))?
            .map_err(TtsError::from)?;
        Ok(TtsResponse::Wav(wav))
    }

    fn synthesize_clone(&self, req: TtsCloneRequest) -> Result<TtsResponse, TtsError> {
        // GPT-SoVITS dedicated clone path goes through Qwen3-TTS's
        // SpeechClone arm in this codebase (see handlers::audio::audio_speech_clone);
        // keep the trait impl honest and route the bytes through the
        // same arm so the inference thread picks the right engine.
        let clone_req = SpeechCloneRequest {
            input: req.input,
            reference_audio: req.reference_audio,
            language: req.language,
            speed: req.speed,
            instruct: req.instruct,
        };
        let (tx, rx) = oneshot::channel();
        self.inference_tx
            .blocking_send(InferenceRequest::Qwen3Tts(InferenceTtsRequest::SpeechClone {
                request: clone_req,
                response_tx: tx,
            }))
            .map_err(|e| TtsError::Backend(format!("inference channel closed: {e}")))?;
        let wav = rx
            .blocking_recv()
            .map_err(|e| TtsError::Backend(format!("inference dropped reply: {e}")))?
            .map_err(TtsError::from)?;
        Ok(TtsResponse::Wav(wav))
    }
}

// ============================================================================
// Selector — resolve ASCEND_TTS_TRANSPORT at startup
// ============================================================================

/// Pick an Ascend TTS backend at startup based on `ASCEND_TTS_TRANSPORT`.
///
/// Values:
///   * `subprocess` (default, and the only one guaranteed to work today)
///     → `AscendSubprocessTts`.
///   * `ffi` → `AscendFfiTts` on Linux with `ascend-tts-ffi` feature;
///     falls back to subprocess with a `tracing::warn!` on Mac or
///     feature-off builds so dev workflows don't break.
///
/// Called once from `main.rs`; the resulting `Arc<dyn TextToSpeech>`
/// lives in `AppState` for the life of the process.
pub fn build_ascend_tts_backend(
    config: Arc<AscendConfig>,
) -> Arc<dyn TextToSpeech> {
    let transport = std::env::var("ASCEND_TTS_TRANSPORT")
        .unwrap_or_else(|_| "subprocess".to_string());
    let transport = transport.trim().to_lowercase();

    match transport.as_str() {
        "ffi" => {
            #[cfg(all(feature = "ascend-tts-ffi", target_os = "linux"))]
            {
                match AscendFfiTts::new(config.clone()) {
                    Ok(b) => {
                        tracing::info!(
                            "TTS backend selected: ascend-ffi (ASCEND_TTS_TRANSPORT=ffi)"
                        );
                        return Arc::new(b);
                    }
                    Err(e) => {
                        tracing::warn!(
                            "ASCEND_TTS_TRANSPORT=ffi but AscendFfiTts init failed: {e}; \
                             falling back to subprocess"
                        );
                    }
                }
            }
            #[cfg(not(all(feature = "ascend-tts-ffi", target_os = "linux")))]
            {
                tracing::warn!(
                    "ASCEND_TTS_TRANSPORT=ffi requested but this build does not include \
                     the ascend-tts-ffi feature on Linux. Falling back to subprocess."
                );
            }
            Arc::new(AscendSubprocessTts::new(config))
        }
        "subprocess" | "" => {
            tracing::info!("TTS backend selected: ascend-subprocess (default)");
            Arc::new(AscendSubprocessTts::new(config))
        }
        other => {
            tracing::warn!(
                "Unknown ASCEND_TTS_TRANSPORT={other:?}; defaulting to subprocess"
            );
            Arc::new(AscendSubprocessTts::new(config))
        }
    }
}

// Silence unused-import warnings on non-FFI builds: `Mutex` is only used
// inside the cfg-gated real impl.
#[cfg(not(all(feature = "ascend-tts-ffi", target_os = "linux")))]
#[allow(dead_code)]
fn _mutex_is_used_on_linux_only(_: Mutex<()>) {}
