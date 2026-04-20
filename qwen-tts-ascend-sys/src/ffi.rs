//! Raw FFI surface.
//!
//! When `ascend-available` is enabled, this module re-exports the
//! bindgen-generated `extern "C"` declarations from `wrapper/qwen_tts_api.h`.
//!
//! When the feature is off (default, Mac dev path), we provide a
//! compile-only stub with the same opaque type and signatures. The stubs
//! are never called — the safe wrapper short-circuits to
//! `TtsError::Unsupported` before reaching them — but the types have to
//! exist so the wrapper module compiles unchanged.

#[cfg(feature = "ascend-available")]
#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    clippy::all
)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[cfg(feature = "ascend-available")]
pub use generated::*;

// ---------------------------------------------------------------------
// Stub surface for non-Ascend targets.
//
// Rationale: keeps `use ffi::qwen_tts_ctx_t` in the wrapper module
// valid on Mac, where the bindgen include is skipped. The function
// decls exist but are never called because `QwenTtsCtx::load` and its
// peers short-circuit to `TtsError::Unsupported` when the feature is
// off.
// ---------------------------------------------------------------------

#[cfg(not(feature = "ascend-available"))]
#[allow(non_camel_case_types, dead_code)]
mod stub {
    use core::ffi::{c_char, c_int, c_void};

    /// Opaque context; stays a ZST-shaped opaque type on stubbed builds.
    #[repr(C)]
    pub struct qwen_tts_ctx {
        _private: [u8; 0],
    }
    pub type qwen_tts_ctx_t = qwen_tts_ctx;

    // The following are declared but never called; marked `unsafe extern`
    // signatures match the real ABI so the wrapper module compiles.
    // We intentionally do NOT emit link directives here — the functions
    // are unreachable on stub builds.
    #[allow(dead_code)]
    extern "C" {
        pub fn qwen_tts_load(
            model_dir: *const c_char,
            tokenizer_dir: *const c_char,
            talker_override: *const c_char,
            cp_override: *const c_char,
            n_gpu_layers: c_int,
            n_threads: c_int,
        ) -> *mut qwen_tts_ctx_t;

        pub fn qwen_tts_free(ctx: *mut qwen_tts_ctx_t);

        pub fn qwen_tts_hidden_size(ctx: *const qwen_tts_ctx_t) -> c_int;
        pub fn qwen_tts_vocab_size(ctx: *const qwen_tts_ctx_t) -> c_int;
        pub fn qwen_tts_has_speaker_encoder(ctx: *const qwen_tts_ctx_t) -> c_int;

        pub fn qwen_tts_text_embed(ctx: *mut qwen_tts_ctx_t, token_id: u32, out: *mut f32);
        pub fn qwen_tts_codec_embed(ctx: *mut qwen_tts_ctx_t, codec_token: u32, out: *mut f32);
        pub fn qwen_tts_codec_head(
            ctx: *mut qwen_tts_ctx_t,
            hidden: *const f32,
            logits_out: *mut f32,
        );
        pub fn qwen_tts_generation_embed(
            ctx: *mut qwen_tts_ctx_t,
            text_embed: *const f32,
            prev_codes: *const u32,
            out: *mut f32,
        );

        pub fn qwen_tts_reset_cache(ctx: *mut qwen_tts_ctx_t);

        pub fn qwen_tts_forward(
            ctx: *mut qwen_tts_ctx_t,
            input_embeds: *const f32,
            seq_len: c_int,
            logits_out: *mut f32,
            hidden_out: *mut f32,
        ) -> c_int;

        pub fn qwen_tts_predict_codes(
            ctx: *mut qwen_tts_ctx_t,
            hidden: *const f32,
            code0: u32,
            codes_out: *mut u32,
        ) -> c_int;

        pub fn qwen_tts_decode_audio(
            ctx: *mut qwen_tts_ctx_t,
            codes: *const u32,
            n_frames: c_int,
            n_groups: c_int,
            audio_out: *mut f32,
            n_samples_out: *mut c_int,
        ) -> c_int;

        pub fn qwen_tts_extract_speaker(
            ctx: *mut qwen_tts_ctx_t,
            audio: *const f32,
            n_samples: c_int,
            sample_rate: c_int,
            embedding_out: *mut f32,
        ) -> c_int;

        // --- B5 high-level ABI (contract §5 B5.1) ---
        pub fn qwen_tts_synthesize(
            ctx: *mut qwen_tts_ctx_t,
            params: *const qwen_tts_synth_params_t,
            pcm_out: *mut *mut f32,
            n_samples_out: *mut c_int,
        ) -> c_int;

        pub fn qwen_tts_pcm_free(pcm: *mut f32);
    }

    /// `qwen_tts_synth_params_t` — stub mirror for non-Ascend builds.
    /// Real layout from contract §5 B5.1; fields must match the C ABI
    /// once the header lands.
    #[repr(C)]
    #[allow(dead_code)]
    pub struct qwen_tts_synth_params_t {
        pub text: *const c_char,
        pub ref_audio_path: *const c_char,
        pub ref_text: *const c_char,
        pub ref_lang: *const c_char,
        pub target_lang: *const c_char,
        pub mode: *const c_char,
        pub speaker: *const c_char,
        pub seed: c_int,
        pub max_tokens: c_int,
        pub temperature: f32,
        pub top_k: c_int,
        pub top_p: f32,
        pub repetition_penalty: f32,
        pub cp_groups: c_int,
        pub cp_layers: c_int,
        pub greedy: c_int,
    }

    // Unused-import silencer: c_void is referenced by the real bindgen
    // output's derived types but not by our stubs; pull it in so editing
    // the stub stays a drop-in match.
    #[allow(dead_code)]
    fn _anchor(_: *mut c_void) {}
}

#[cfg(not(feature = "ascend-available"))]
pub use stub::*;
