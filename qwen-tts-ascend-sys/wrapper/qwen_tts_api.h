/*
 * Pinned from: OminiX-Ascend @ 12405a5251346d9568116e801c88b22bced661e8
 *              tools/qwen_tts/qwen_tts_api.h (working-tree state as of
 *              2026-04-19 post-B5.1 landing; upstream commit for the B5
 *              symbols pending — Agent X is still staging).
 * Content SHA-256 (upstream, before this pin header):
 *   042025b6979ad2096990e1f42f5253a46fb6dfc90de258abfe3ca022c7d892e9
 *
 * To re-vendor after an upstream change:
 *   1. Copy OminiX-Ascend/tools/qwen_tts/qwen_tts_api.h over this file.
 *   2. Re-prepend this block with updated commit + sha256.
 *   3. Bump qwen-tts-ascend-sys crate version.
 *
 * build.rs does NOT currently assert the content hash (no in-tree check
 * yet); the hash above is informational and must be re-recorded by the
 * agent doing a header refresh.
 */
/**
 * qwen_tts_api.h — Minimal C API for Qwen3-TTS compute primitives.
 *
 * Exposes embedding lookup, transformer forward pass, code prediction,
 * speech decoding, and speaker encoding as C functions.
 * The generation loop logic (sampling, prefill building, CustomVoice/x-vector)
 * lives in Rust (qwen3-tts-core).
 */

#ifndef QWEN_TTS_API_H
#define QWEN_TTS_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque context handle */
typedef struct qwen_tts_ctx qwen_tts_ctx_t;

/**
 * Load all TTS models from a GGUF directory.
 *
 * model_dir: directory containing qwen_tts_talker.gguf, qwen_tts_talker_llama*.gguf,
 *            qwen_tts_code_predictor.gguf, qwen_tts_tokenizer_*.gguf,
 *            qwen_tts_speaker_encoder.gguf (optional), vocab.json, merges.txt
 * tokenizer_dir: directory with vocab.json + merges.txt (NULL = same as model_dir)
 * talker_override: override talker llama GGUF path (NULL = auto from model_dir)
 * cp_override: override code predictor llama GGUF path (NULL = auto)
 * n_gpu_layers: layers to offload to NPU (29 = all for 1.7B)
 * n_threads: CPU threads
 *
 * Returns: context handle, or NULL on failure.
 */
qwen_tts_ctx_t* qwen_tts_load(
    const char* model_dir,
    const char* tokenizer_dir,
    const char* talker_override,
    const char* cp_override,
    int n_gpu_layers,
    int n_threads
);

/** Free all resources. */
void qwen_tts_free(qwen_tts_ctx_t* ctx);

/** Get model hidden size (typically 2048). */
int qwen_tts_hidden_size(const qwen_tts_ctx_t* ctx);

/** Get codec vocabulary size (typically 3072). */
int qwen_tts_vocab_size(const qwen_tts_ctx_t* ctx);

/** Check if speaker encoder is loaded (Base model only). */
int qwen_tts_has_speaker_encoder(const qwen_tts_ctx_t* ctx);

/* ========================================================================== */
/* Embedding operations                                                       */
/* ========================================================================== */

/**
 * Compute text_proj(text_embed(token_id)).
 * out: buffer of size hidden_size.
 */
void qwen_tts_text_embed(qwen_tts_ctx_t* ctx, uint32_t token_id, float* out);

/**
 * Lookup raw codec embedding (no projection).
 * out: buffer of size hidden_size.
 */
void qwen_tts_codec_embed(qwen_tts_ctx_t* ctx, uint32_t codec_token, float* out);

/**
 * Apply codec_head to hidden state → logits.
 * hidden: [hidden_size], logits_out: [vocab_size]
 */
void qwen_tts_codec_head(qwen_tts_ctx_t* ctx, const float* hidden, float* logits_out);

/**
 * Compute generation embedding: sum of 16 codec group embeddings for prev frame + text.
 * text_embed: [hidden_size] (pre-computed text_proj)
 * prev_codes: [16] codec values from previous frame
 * out: [hidden_size]
 */
void qwen_tts_generation_embed(
    qwen_tts_ctx_t* ctx,
    const float* text_embed,
    const uint32_t* prev_codes,
    float* out
);

/* ========================================================================== */
/* Transformer forward pass                                                   */
/* ========================================================================== */

/**
 * Reset KV cache (call before each new generation).
 */
void qwen_tts_reset_cache(qwen_tts_ctx_t* ctx);

/**
 * Forward pass through the 28-layer transformer backbone.
 *
 * input_embeds: [seq_len * hidden_size] flat f32
 * seq_len: number of positions
 * logits_out: [vocab_size] logits from LAST position (after codec_head)
 * hidden_out: [hidden_size] hidden state from LAST position
 *
 * Returns 0 on success, non-zero on error.
 */
int qwen_tts_forward(
    qwen_tts_ctx_t* ctx,
    const float* input_embeds,
    int seq_len,
    float* logits_out,
    float* hidden_out
);

/* ========================================================================== */
/* Code prediction (codebooks 1-15)                                          */
/* ========================================================================== */

/**
 * Generate sub-codes (codebooks 1-15) from hidden state + code0.
 *
 * hidden: [hidden_size] from transformer output
 * code0: group 0 token
 * codes_out: buffer of size 15 (groups 1-15)
 *
 * Returns 0 on success.
 */
int qwen_tts_predict_codes(
    qwen_tts_ctx_t* ctx,
    const float* hidden,
    uint32_t code0,
    uint32_t* codes_out
);

/* ========================================================================== */
/* Speech tokenizer decoder (codes → audio)                                  */
/* ========================================================================== */

/**
 * Decode codec frames to audio waveform.
 *
 * codes: [n_frames * n_groups] flat u32 (row-major: frame0_g0, frame0_g1, ..., frame1_g0, ...)
 * n_frames: number of codec frames
 * n_groups: codebook groups (16)
 * audio_out: output buffer (caller allocates, recommend n_frames * 1920 floats)
 * n_samples_out: actual number of audio samples written
 *
 * Returns 0 on success.
 */
int qwen_tts_decode_audio(
    qwen_tts_ctx_t* ctx,
    const uint32_t* codes,
    int n_frames,
    int n_groups,
    float* audio_out,
    int* n_samples_out
);

/* ========================================================================== */
/* Speaker encoder (reference audio → embedding)                             */
/* ========================================================================== */

/**
 * Extract speaker embedding from reference audio.
 *
 * audio: f32 samples at sample_rate Hz
 * n_samples: number of samples
 * sample_rate: audio sample rate (24000 recommended)
 * embedding_out: buffer of size hidden_size (typically 2048)
 *
 * Returns 0 on success, -1 if speaker encoder not loaded.
 */
int qwen_tts_extract_speaker(
    qwen_tts_ctx_t* ctx,
    const float* audio,
    int n_samples,
    int sample_rate,
    float* embedding_out
);

/* ========================================================================== */
/* High-level one-shot synthesis (Ascend API Bridge Contract §5 B5)           */
/* ========================================================================== */

/**
 * Parameters for qwen_tts_synthesize().
 *
 * Mirrors the fields of QwenTTSParams (see qwen_tts.h) actually consumed by
 * QwenTTS::generate() / generate_xvec() / generate_customvoice(). Callers
 * construct, fill, and pass by pointer; the library never retains it past
 * the synthesize() call.
 *
 * Zero-defaulted sampling fields use the same defaults as the qwen_tts
 * CLI / Python reference (documented per-field below). Set any field to
 * its "use default" sentinel to fall through.
 */
typedef struct {
    const char* text;               /* target text (UTF-8, non-null)            */
    const char* ref_audio_path;     /* path to ref .wav (NULL if mode != "icl") */
    const char* ref_text;           /* reference transcript (NULL if not ICL)   */
    const char* ref_lang;           /* "English" | "Chinese" | etc              */
    const char* target_lang;        /* same set                                 */
    const char* mode;               /* "icl" | "xvec" | "customvoice"           */
    const char* speaker;            /* for customvoice mode (NULL otherwise)    */
    int         seed;               /* sampling seed (0 = non-deterministic)    */
    int         max_tokens;         /* 0 = default 2048                         */
    float       temperature;        /* 0 = default 0.9                          */
    int         top_k;              /* 0 = default 50, -1 = disabled            */
    float       top_p;              /* 0 = default 1.0                          */
    float       repetition_penalty; /* 0 = default 1.05                         */
    int         cp_groups;          /* 0 = default (all 15)                     */
    int         cp_layers;          /* 0 = default (all 5)                      */
    int         greedy;             /* 0 = sample, non-zero = greedy            */
} qwen_tts_synth_params_t;

/**
 * One-shot text-to-speech synthesis.
 *
 * Dispatches by params->mode to QwenTTS::generate() (ICL),
 * generate_xvec(), or generate_customvoice(). The library allocates the
 * output PCM buffer via malloc() (size is unknown up front); the caller
 * MUST release it with qwen_tts_pcm_free().
 *
 * pcm_out:       [out] pointer receiving the malloc()'d f32 buffer (24kHz mono)
 * n_samples_out: [out] number of float samples written
 *
 * Returns 0 on success. On any error, *pcm_out is set to NULL and
 * *n_samples_out is set to 0. Error codes:
 *   -1: ctx or params was NULL
 *   -2: unknown mode
 *   -3: generation failed inside QwenTTS::generate*
 */
int qwen_tts_synthesize(
    qwen_tts_ctx_t*                   ctx,
    const qwen_tts_synth_params_t*    params,
    float**                           pcm_out,
    int*                              n_samples_out
);

/**
 * Release a PCM buffer returned by qwen_tts_synthesize().
 *
 * Equivalent to free(pcm). Safe to call with NULL (no-op).
 */
void qwen_tts_pcm_free(float* pcm);

#ifdef __cplusplus
}
#endif

#endif /* QWEN_TTS_API_H */
