//! # qwen-tts-ascend-sys
//!
//! Rust FFI bindings for `libqwen_tts_api.so` — the Qwen3-TTS compute ABI
//! exported by OminiX-Ascend for Ascend 910 NPUs.
//!
//! See `ASCEND_API_BRIDGE_CONTRACT.md` §5 B2 for the milestone spec.
//!
//! ## Feature flag: `ascend-available`
//!
//! The real FFI surface is gated on the `ascend-available` feature so
//! that `cargo check` on Mac dev hosts works without the `.so` being
//! present. When the feature is off the crate still compiles and all
//! `QwenTtsCtx` methods return `TtsError::Unsupported("…")`. This keeps
//! the wrapper-type surface stable across platforms and lets upstream
//! code write `cfg`-free call sites that degrade cleanly.
//!
//! ## Thread safety
//!
//! `QwenTtsCtx` is `Send` (it can move between threads between calls)
//! but **not** `Sync`: per the OminiX-Ascend native TTS finding, a
//! single handle is not safe to call concurrently because its KV cache
//! is mutated by `qwen_tts_forward`. Callers who want parallelism must
//! allocate one handle per worker thread; callers who share a handle
//! must serialize calls via a `Mutex` at a higher layer.

#![allow(clippy::missing_safety_doc)]

pub mod ffi;
pub mod wrapper;

pub use wrapper::{QwenTtsCtx, SynthParams, TtsError};
