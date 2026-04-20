//! Safe(-ish) Rust wrapper over the `qwen_tts_api` C ABI.
//!
//! All raw pointers are confined to this module. Callers get:
//!   * RAII: `Drop` frees the context via `qwen_tts_free`.
//!   * Checked buffer sizes (we look up `hidden_size` / `vocab_size`
//!     from the loaded model and reject wrong-sized slices).
//!   * `thiserror`-typed errors instead of `-1` sentinels.
//!   * `Send` but not `Sync` (see struct-level safety note).
//!
//! The safety invariants for each `unsafe` block are documented inline.

#[cfg(feature = "ascend-available")]
use std::ffi::CString;
use std::os::raw::c_int;
use std::path::Path;

use thiserror::Error;

use crate::ffi;

/// All failure modes this wrapper can produce.
#[derive(Debug, Error)]
pub enum TtsError {
    /// `qwen_tts_load` returned NULL, or input path-strings contained
    /// interior NUL bytes.
    #[error("qwen_tts_load failed: {0}")]
    LoadFailed(String),

    /// `qwen_tts_forward` returned a non-zero status code.
    #[error("qwen_tts_forward failed with code {0}")]
    ForwardFailed(c_int),

    /// `qwen_tts_predict_codes` returned a non-zero status code.
    #[error("qwen_tts_predict_codes failed with code {0}")]
    PredictFailed(c_int),

    /// `qwen_tts_decode_audio` returned a non-zero status code.
    #[error("qwen_tts_decode_audio failed with code {0}")]
    DecodeFailed(c_int),

    /// `qwen_tts_extract_speaker` returned a non-zero status code
    /// (-1 == speaker encoder not loaded, per the C header).
    #[error("qwen_tts_extract_speaker failed with code {0}")]
    SpeakerExtractFailed(c_int),

    /// The requested operation is not supported on this build target
    /// (e.g. Mac dev host without the `ascend-available` feature, or a
    /// model loaded without a speaker encoder).
    #[error("operation unsupported on this target: {0}")]
    Unsupported(&'static str),
}

/// Safe RAII handle over a loaded Qwen-TTS context.
///
/// ## Invariants
///
/// * `raw` is either null (never, post-construction — we reject NULL in
///   `load`) or a valid pointer obtained from `qwen_tts_load` that has
///   not yet been passed to `qwen_tts_free`.
/// * `hidden_size` and `vocab_size` are cached at load time; the C ABI
///   contract is that they don't change across the life of the handle.
///
/// ## Thread safety
///
/// `QwenTtsCtx` is `Send` (no thread-local state is parked in the
/// pointer; ownership can move across threads). It is explicitly
/// **not** `Sync`: `qwen_tts_forward` mutates the KV cache inside the
/// engine and two threads touching the same handle concurrently will
/// corrupt it (documented risk in contract §7). Share via
/// `Arc<Mutex<QwenTtsCtx>>` or allocate one handle per worker.
pub struct QwenTtsCtx {
    // `raw` is unused on the stub (no-feature) build because we never
    // actually call into the C side; silence the unused-field warning
    // rather than gating the field itself, which would churn the Drop
    // impl and method bodies.
    #[allow(dead_code)]
    raw: *mut ffi::qwen_tts_ctx_t,
    hidden_size: usize,
    vocab_size: usize,
}

// Safety: the underlying handle can safely move across threads as long
// as only one thread at a time calls into it. Our public API takes
// `&mut self` on every mutating method, so Rust's borrow checker
// enforces the single-caller rule for a given handle statically. We do
// NOT impl Sync: two `&QwenTtsCtx` shared across threads would allow
// concurrent calls to read-only queries, but more importantly would
// invite users to smuggle `&mut` access via interior mutability on top,
// which would violate the engine's no-concurrent-call rule. Keeping
// Sync unimplemented forces the Mutex wrapper at the use site.
unsafe impl Send for QwenTtsCtx {}

/// Number of codec groups per frame (fixed by the model architecture).
pub const N_CODEC_GROUPS: usize = 16;

impl QwenTtsCtx {
    /// Load all TTS models from a GGUF directory.
    ///
    /// See `qwen_tts_api.h::qwen_tts_load` for argument semantics.
    /// `tokenizer_dir`, `talker_override`, and `cp_override` accept
    /// `None` to pass NULL (auto-detect from `model_dir`).
    pub fn load(
        model_dir: &Path,
        tokenizer_dir: Option<&Path>,
        talker_override: Option<&Path>,
        cp_override: Option<&Path>,
        n_gpu_layers: i32,
        n_threads: i32,
    ) -> Result<Self, TtsError> {
        #[cfg(not(feature = "ascend-available"))]
        {
            let _ = (
                model_dir,
                tokenizer_dir,
                talker_override,
                cp_override,
                n_gpu_layers,
                n_threads,
            );
            Err(TtsError::Unsupported(
                "qwen-tts-ascend-sys built without `ascend-available` feature",
            ))
        }

        #[cfg(feature = "ascend-available")]
        {
            fn to_cstring(p: &Path) -> Result<CString, TtsError> {
                CString::new(p.to_string_lossy().into_owned().into_bytes())
                    .map_err(|e| TtsError::LoadFailed(format!("path has NUL byte: {e}")))
            }

            let c_model = to_cstring(model_dir)?;
            let c_tokenizer = tokenizer_dir.map(to_cstring).transpose()?;
            let c_talker = talker_override.map(to_cstring).transpose()?;
            let c_cp = cp_override.map(to_cstring).transpose()?;

            // Safety: all pointers are either NULL (Option::None) or
            // valid CString buffers live for the duration of the call.
            // `qwen_tts_load` takes ownership of nothing — it copies what
            // it needs — so dropping the CStrings after the call is safe.
            let raw = unsafe {
                ffi::qwen_tts_load(
                    c_model.as_ptr(),
                    c_tokenizer.as_ref().map(|s| s.as_ptr()).unwrap_or(std::ptr::null()),
                    c_talker.as_ref().map(|s| s.as_ptr()).unwrap_or(std::ptr::null()),
                    c_cp.as_ref().map(|s| s.as_ptr()).unwrap_or(std::ptr::null()),
                    n_gpu_layers as c_int,
                    n_threads as c_int,
                )
            };

            if raw.is_null() {
                return Err(TtsError::LoadFailed(format!(
                    "qwen_tts_load returned NULL for model_dir={}",
                    model_dir.display()
                )));
            }

            // Safety: raw is non-null and freshly returned from the C
            // side; the C contract guarantees these query functions are
            // callable on a just-loaded handle.
            let hidden_size = unsafe { ffi::qwen_tts_hidden_size(raw) } as usize;
            let vocab_size = unsafe { ffi::qwen_tts_vocab_size(raw) } as usize;

            Ok(Self { raw, hidden_size, vocab_size })
        }
    }

    /// Model hidden size (typically 2048). Cached at load time.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Codec vocabulary size (typically 3072). Cached at load time.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// `true` if this context was loaded with a speaker encoder.
    pub fn has_speaker_encoder(&self) -> bool {
        #[cfg(not(feature = "ascend-available"))]
        {
            false
        }
        #[cfg(feature = "ascend-available")]
        {
            // Safety: `raw` is a live handle per the struct invariants.
            unsafe { ffi::qwen_tts_has_speaker_encoder(self.raw) != 0 }
        }
    }

    /// Reset the transformer KV cache. Call before each new generation.
    pub fn reset_cache(&mut self) {
        #[cfg(feature = "ascend-available")]
        {
            // Safety: `raw` is live and `&mut self` serializes callers.
            unsafe { ffi::qwen_tts_reset_cache(self.raw) }
        }
    }

    /// Compute `text_proj(text_embed(token_id))`; writes `hidden_size`
    /// floats into `out`.
    pub fn text_embed(&self, token_id: u32, out: &mut [f32]) -> Result<(), TtsError> {
        self.check_buffer(out.len(), self.hidden_size, "text_embed out")?;
        #[cfg(feature = "ascend-available")]
        {
            // Safety: buffer length was checked above; `raw` is live.
            // C ABI is `mut` on ctx but this function mutates no engine
            // state beyond internal scratch, so a `&self` receiver is OK.
            unsafe { ffi::qwen_tts_text_embed(self.raw, token_id, out.as_mut_ptr()) }
            Ok(())
        }
        #[cfg(not(feature = "ascend-available"))]
        {
            let _ = token_id;
            Err(TtsError::Unsupported("text_embed needs `ascend-available`"))
        }
    }

    /// Look up the raw codec embedding for `codec_token` (no projection).
    pub fn codec_embed(&self, codec_token: u32, out: &mut [f32]) -> Result<(), TtsError> {
        self.check_buffer(out.len(), self.hidden_size, "codec_embed out")?;
        #[cfg(feature = "ascend-available")]
        {
            // Safety: buffer length was checked above; `raw` is live.
            unsafe { ffi::qwen_tts_codec_embed(self.raw, codec_token, out.as_mut_ptr()) }
            Ok(())
        }
        #[cfg(not(feature = "ascend-available"))]
        {
            let _ = codec_token;
            Err(TtsError::Unsupported("codec_embed needs `ascend-available`"))
        }
    }

    /// Apply `codec_head` to a hidden state and produce logits.
    pub fn codec_head(&self, hidden: &[f32], logits_out: &mut [f32]) -> Result<(), TtsError> {
        self.check_buffer(hidden.len(), self.hidden_size, "codec_head hidden")?;
        self.check_buffer(logits_out.len(), self.vocab_size, "codec_head logits_out")?;
        #[cfg(feature = "ascend-available")]
        {
            // Safety: both buffers' lengths validated above.
            unsafe {
                ffi::qwen_tts_codec_head(self.raw, hidden.as_ptr(), logits_out.as_mut_ptr())
            }
            Ok(())
        }
        #[cfg(not(feature = "ascend-available"))]
        {
            Err(TtsError::Unsupported("codec_head needs `ascend-available`"))
        }
    }

    /// Build a generation embedding: sum of 16 codec group embeddings
    /// for the previous frame plus `text_embed`.
    ///
    /// `text_embed` must be `hidden_size`, `prev_codes` must be 16.
    pub fn generation_embed(
        &self,
        text_embed: &[f32],
        prev_codes: &[u32],
        out: &mut [f32],
    ) -> Result<(), TtsError> {
        self.check_buffer(text_embed.len(), self.hidden_size, "generation_embed text_embed")?;
        self.check_buffer(prev_codes.len(), N_CODEC_GROUPS, "generation_embed prev_codes")?;
        self.check_buffer(out.len(), self.hidden_size, "generation_embed out")?;
        #[cfg(feature = "ascend-available")]
        {
            // Safety: all three buffers' lengths validated above.
            unsafe {
                ffi::qwen_tts_generation_embed(
                    self.raw,
                    text_embed.as_ptr(),
                    prev_codes.as_ptr(),
                    out.as_mut_ptr(),
                )
            }
            Ok(())
        }
        #[cfg(not(feature = "ascend-available"))]
        {
            Err(TtsError::Unsupported("generation_embed needs `ascend-available`"))
        }
    }

    /// Forward pass through the 28-layer backbone. `input_embeds` must
    /// be exactly `seq_len * hidden_size` long; writes last-position
    /// logits + hidden out buffers.
    pub fn forward(
        &mut self,
        input_embeds: &[f32],
        seq_len: usize,
        logits_out: &mut [f32],
        hidden_out: &mut [f32],
    ) -> Result<(), TtsError> {
        let expected_embed = seq_len.checked_mul(self.hidden_size).ok_or_else(|| {
            TtsError::Unsupported("seq_len * hidden_size overflows usize")
        })?;
        self.check_buffer(input_embeds.len(), expected_embed, "forward input_embeds")?;
        self.check_buffer(logits_out.len(), self.vocab_size, "forward logits_out")?;
        self.check_buffer(hidden_out.len(), self.hidden_size, "forward hidden_out")?;

        #[cfg(feature = "ascend-available")]
        {
            // Safety: all three buffers validated; `&mut self` ensures
            // no concurrent caller is touching the KV cache.
            let rc = unsafe {
                ffi::qwen_tts_forward(
                    self.raw,
                    input_embeds.as_ptr(),
                    seq_len as c_int,
                    logits_out.as_mut_ptr(),
                    hidden_out.as_mut_ptr(),
                )
            };
            if rc == 0 {
                Ok(())
            } else {
                Err(TtsError::ForwardFailed(rc))
            }
        }
        #[cfg(not(feature = "ascend-available"))]
        {
            let _ = (input_embeds, seq_len, logits_out, hidden_out);
            Err(TtsError::Unsupported("forward needs `ascend-available`"))
        }
    }

    /// Predict codebook groups 1-15 from a hidden state + group-0 token.
    /// `codes_out` must be length 15.
    pub fn predict_codes(
        &mut self,
        hidden: &[f32],
        code0: u32,
        codes_out: &mut [u32],
    ) -> Result<(), TtsError> {
        self.check_buffer(hidden.len(), self.hidden_size, "predict_codes hidden")?;
        self.check_buffer(codes_out.len(), N_CODEC_GROUPS - 1, "predict_codes codes_out")?;

        #[cfg(feature = "ascend-available")]
        {
            // Safety: buffer lengths validated; exclusive access via &mut self.
            let rc = unsafe {
                ffi::qwen_tts_predict_codes(
                    self.raw,
                    hidden.as_ptr(),
                    code0,
                    codes_out.as_mut_ptr(),
                )
            };
            if rc == 0 {
                Ok(())
            } else {
                Err(TtsError::PredictFailed(rc))
            }
        }
        #[cfg(not(feature = "ascend-available"))]
        {
            let _ = (hidden, code0, codes_out);
            Err(TtsError::Unsupported("predict_codes needs `ascend-available`"))
        }
    }

    /// Decode `n_frames` of codec tokens to audio samples.
    ///
    /// `codes` must have exactly `n_frames * N_CODEC_GROUPS` elements.
    /// `audio_out` should be sized generously (header recommends
    /// `n_frames * 1920`); the actual written-sample count is returned.
    pub fn decode_audio(
        &mut self,
        codes: &[u32],
        n_frames: usize,
        audio_out: &mut [f32],
    ) -> Result<usize, TtsError> {
        let expected_codes = n_frames.checked_mul(N_CODEC_GROUPS).ok_or_else(|| {
            TtsError::Unsupported("n_frames * N_CODEC_GROUPS overflows usize")
        })?;
        self.check_buffer(codes.len(), expected_codes, "decode_audio codes")?;

        #[cfg(feature = "ascend-available")]
        {
            let mut n_samples_out: c_int = 0;
            // Safety: codes length validated; audio_out slice is mut so
            // we can write up to its length; C writes at most
            // `n_frames * 1920` samples per header contract.
            let rc = unsafe {
                ffi::qwen_tts_decode_audio(
                    self.raw,
                    codes.as_ptr(),
                    n_frames as c_int,
                    N_CODEC_GROUPS as c_int,
                    audio_out.as_mut_ptr(),
                    &mut n_samples_out,
                )
            };
            if rc == 0 {
                let written = n_samples_out.max(0) as usize;
                if written > audio_out.len() {
                    // Shouldn't happen if caller sized per header spec,
                    // but guard against buffer-overrun claims.
                    return Err(TtsError::DecodeFailed(-999));
                }
                Ok(written)
            } else {
                Err(TtsError::DecodeFailed(rc))
            }
        }
        #[cfg(not(feature = "ascend-available"))]
        {
            let _ = (codes, n_frames, audio_out);
            Err(TtsError::Unsupported("decode_audio needs `ascend-available`"))
        }
    }

    /// Extract a speaker embedding from reference audio. Requires the
    /// model to have been loaded with a speaker encoder (Base model).
    pub fn extract_speaker(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        embedding_out: &mut [f32],
    ) -> Result<(), TtsError> {
        if !self.has_speaker_encoder() {
            return Err(TtsError::Unsupported(
                "this context has no speaker encoder (CustomVoice model?)",
            ));
        }
        self.check_buffer(embedding_out.len(), self.hidden_size, "extract_speaker embedding_out")?;

        #[cfg(feature = "ascend-available")]
        {
            // Safety: embedding_out length validated; audio is read-only.
            let rc = unsafe {
                ffi::qwen_tts_extract_speaker(
                    self.raw,
                    audio.as_ptr(),
                    audio.len() as c_int,
                    sample_rate as c_int,
                    embedding_out.as_mut_ptr(),
                )
            };
            if rc == 0 {
                Ok(())
            } else {
                Err(TtsError::SpeakerExtractFailed(rc))
            }
        }
        #[cfg(not(feature = "ascend-available"))]
        {
            let _ = (audio, sample_rate, embedding_out);
            Err(TtsError::Unsupported("extract_speaker needs `ascend-available`"))
        }
    }

    fn check_buffer(
        &self,
        actual: usize,
        expected: usize,
        label: &'static str,
    ) -> Result<(), TtsError> {
        if actual == expected {
            Ok(())
        } else {
            // Reuse Unsupported for shape errors to keep the error enum
            // minimal per the contract (avoids adding a BadBufferSize
            // variant that downstream callers would have to match).
            Err(TtsError::Unsupported(label))
        }
    }
}

impl Drop for QwenTtsCtx {
    fn drop(&mut self) {
        #[cfg(feature = "ascend-available")]
        {
            if !self.raw.is_null() {
                // Safety: struct invariant says `raw` is a live handle
                // obtained from qwen_tts_load and not yet freed. We null
                // it out after the call so a double-drop is a no-op.
                unsafe { ffi::qwen_tts_free(self.raw) };
                self.raw = std::ptr::null_mut();
            }
        }
    }
}

// ---------------------------------------------------------------------
// Tests
//
// The RAII lifecycle test only runs when an actual Ascend library is
// available. On Mac `cargo test` just compiles the stub path and skips.
// ---------------------------------------------------------------------

#[cfg(all(test, feature = "ascend-available", target_os = "linux"))]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_dir_from_env() -> Option<PathBuf> {
        std::env::var_os("ASCEND_TTS_MODEL_DIR").map(PathBuf::from)
    }

    #[test]
    fn raii_lifecycle() {
        let dir = match model_dir_from_env() {
            Some(p) if p.exists() => p,
            _ => {
                eprintln!("skipping: ASCEND_TTS_MODEL_DIR not set or missing");
                return;
            }
        };

        let ctx = QwenTtsCtx::load(&dir, None, None, None, 29, 8)
            .expect("qwen_tts_load must succeed when the .so is present");
        assert!(ctx.hidden_size() > 0);
        assert!(ctx.vocab_size() > 0);
        // Drop runs here; no explicit free.
    }
}
