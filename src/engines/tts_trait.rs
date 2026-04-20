//! `TextToSpeech` trait — shared v1 surface for all TTS backends.
//!
//! Contract: `ASCEND_API_BRIDGE_CONTRACT.md` §5 B3.
//!
//! Rationale for placement: this file lives under `src/engines/` next to
//! the existing backend implementations (`ascend.rs`, `qwen3_tts.rs`,
//! `tts.rs`) so that a reader looking at one backend sees the trait in
//! the same directory. A top-level `src/tts/` module was considered and
//! rejected — it would have duplicated the `engines/` layout without
//! buying us anything; we'd still want `engines::ascend` for the non-TTS
//! engines (ASR, LLM, VLM, Image) on that backend.
//!
//! ## Response shape
//!
//! The contract sketches `TtsResponse = { pcm: Vec<i16 or f32>,
//! sample_rate: u32 }`. In practice the four backends disagree on what
//! they return today:
//!   * Ascend subprocess (`qwen_tts` binary) → WAV bytes on disk
//!     (RIFF + 16-bit PCM payload).
//!   * Ascend FFI (B2 wrapper) → `Vec<f32>` samples + rate 24000.
//!   * Qwen3-TTS MLX → `Vec<u8>` WAV via `hound` (wrapping i16 samples).
//!   * GPT-SoVITS MLX → `Vec<u8>` WAV via `hound`.
//!
//! Rather than forcing all four to re-encode/decode on every call (which
//! would break "subprocess path byte-identical" in §5.3.4), the enum
//! below keeps both representations explicit. Callers that need WAV
//! bytes can ask for them; callers that want raw PCM for streaming can
//! ask for that. The handler layer bridges.
//!
//! ## Clone request shape
//!
//! Mirrors `SpeechCloneRequest` (raw reference-audio bytes + text + lang
//! + speed + instruct), so wrapping the existing Ascend subprocess flow
//! is mechanical.

// `ascend_config` is wired to `None` in `main.rs` today (the whole Ascend
// block is commented out), so the trait impls are only exercised on
// Ascend-enabled builds. Allow dead-code warnings at the module level
// rather than pepper every item.
#![allow(dead_code)]

use thiserror::Error;

// ============================================================================
// Request / response types
// ============================================================================

/// Preset-voice TTS request — matches the shape already flowing through
/// `AscendTtsEngine::synthesize` (`&SpeechRequest`) and the MLX
/// `synthesize_one_sentence` path. Everything a backend needs to render
/// one utterance with a named voice.
#[derive(Debug, Clone)]
pub struct TtsRequest {
    /// Text to synthesize.
    pub input: String,
    /// Voice name. Backends resolve this against their own voice catalog
    /// (reference-audio file for Ascend; preset pickle for MLX).
    pub voice: String,
    /// Target language: `chinese`, `english`, `japanese`, `korean`, …
    /// Passed through verbatim — backends may accept different casings.
    pub language: String,
    /// Speaking speed multiplier. `1.0` = normal. Many backends ignore
    /// this today; kept in the signature so they can start honoring it
    /// without an API break.
    pub speed: f32,
    /// Optional natural-language style/emotion instruction.
    pub instruct: Option<String>,
}

/// Voice-cloning TTS request — raw reference-audio bytes (WAV/MP3/OGG)
/// plus the text to render in the cloned voice.
#[derive(Debug, Clone)]
pub struct TtsCloneRequest {
    /// Text to synthesize.
    pub input: String,
    /// Raw reference-audio bytes (WAV/MP3/OGG — backend decodes).
    pub reference_audio: Vec<u8>,
    /// Target language (same semantics as `TtsRequest::language`).
    pub language: String,
    /// Speaking speed multiplier.
    pub speed: f32,
    /// Optional natural-language style/emotion instruction.
    pub instruct: Option<String>,
}

/// TTS output — either a fully-formed WAV container (subprocess path
/// and the current MLX backends write WAV via `hound`), or raw PCM
/// samples if the backend can cheaply hand them over.
#[derive(Debug, Clone)]
pub enum TtsResponse {
    /// Complete WAV file (RIFF header + PCM payload).
    Wav(Vec<u8>),
    /// Raw 16-bit signed-LE PCM samples + sample rate.
    Pcm { samples: Vec<i16>, sample_rate: u32 },
}

impl TtsResponse {
    /// Consume self and produce WAV bytes. Zero-copy for `Wav`; encodes
    /// `Pcm` via the existing `pcm_to_wav` helper in `handlers::audio`
    /// is not called here to avoid a cycle — callers re-encode if they
    /// need it (kept out of the trait surface to keep impls simple).
    pub fn into_wav_if_already(self) -> Option<Vec<u8>> {
        match self {
            TtsResponse::Wav(v) => Some(v),
            TtsResponse::Pcm { .. } => None,
        }
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Unified error type for the TTS trait. Handler code maps these to
/// HTTP status codes.
#[derive(Debug, Error)]
pub enum TtsError {
    /// Backend doesn't support the requested capability (e.g. cloning
    /// on a preset-only voice model, or FFI on a Mac dev host).
    #[error("unsupported: {0}")]
    Unsupported(&'static str),

    /// Backend is reachable but rejected the request payload (bad voice
    /// name, unparseable language code, etc.).
    #[error("bad request: {0}")]
    BadRequest(String),

    /// The underlying subprocess / FFI call / channel failed. Opaque
    /// string so we can wrap `eyre::Report`, `std::io::Error`,
    /// `qwen_tts_ascend_sys::TtsError`, and `oneshot::RecvError` without
    /// a variant explosion.
    #[error("backend failure: {0}")]
    Backend(String),
}

impl From<eyre::Report> for TtsError {
    fn from(e: eyre::Report) -> Self {
        TtsError::Backend(format!("{e:#}"))
    }
}

// ============================================================================
// The trait
// ============================================================================

/// Unified v1 TTS backend interface. Must be object-safe (`dyn
/// TextToSpeech`) so handlers can hold `Arc<dyn TextToSpeech>` from app
/// state — hence `&self` receivers and no associated types.
///
/// Non-thread-safe backends (Ascend FFI, whose handle is `!Sync`) must
/// guard their state internally (typically via `Mutex`) so this trait's
/// `&self` contract holds from the outside.
pub trait TextToSpeech: Send + Sync {
    /// Identifier for logs and `/v1/models` reporting. Stable across a
    /// process; not a user-facing string.
    fn backend_name(&self) -> &'static str;

    /// Whether `synthesize_clone` will do something other than return
    /// `Unsupported`. The default `synthesize_clone` impl returns
    /// `Unsupported`; override both together.
    fn supports_clone(&self) -> bool;

    /// Render one utterance with a named preset voice.
    fn synthesize(&self, req: TtsRequest) -> Result<TtsResponse, TtsError>;

    /// Render one utterance in a voice cloned from reference audio.
    /// Default implementation rejects — backends that support cloning
    /// must override and also return `true` from `supports_clone`.
    fn synthesize_clone(
        &self,
        _req: TtsCloneRequest,
    ) -> Result<TtsResponse, TtsError> {
        Err(TtsError::Unsupported(
            "clone not supported by this backend",
        ))
    }
}
