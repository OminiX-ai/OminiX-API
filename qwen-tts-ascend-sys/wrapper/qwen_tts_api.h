/*
 * Pinned from: OminiX-Ascend @ d5a1e761 (merge: take origin/main for tools/qwen_tts/)
 * Upstream content SHA-256: 76fd5e19aee46df97520024e308ff9dcd6f64f3614773de1d37e0f7050f88bac
 * Upstream path:  tools/qwen_tts/qwen_tts_api.h
 *
 * v1.2 shrunk the surface from 16 symbols to 7 after upstream's
 * QwenTTS::generate() unification removed the TalkerLLM friend accessors
 * the 9 fine-grained primitives depended on. bindings.rs now generates
 * 7 extern declarations; wrapper.rs methods for the removed primitives
 * were dropped in the same PR.
 */

/**
 * qwen_tts_api.h — C API for Qwen3-TTS one-shot synthesis.
 *
 * v1.2 (2026-04-20): fine-grained primitive ABI (text_embed, forward,
 * predict_codes, decode_audio, extract_speaker, reset_cache, codec_head,
 * codec_embed, generation_embed) dropped in favour of the single
 * high-level `qwen_tts_synthesize` path. The primitives required
 * friend-accessor methods on TalkerLLM that upstream's unified
 * `QwenTTS::generate()` refactor removed; no caller of the primitives
 * exists in this tree. Bindings consumers (qwen-tts-ascend-sys) call
 * `qwen_tts_synthesize` exclusively. Removed symbols stay in the version
 * script as historical docs in the .version file but are not exported.
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
/* High-level one-shot synthesis (Ascend API Bridge Contract §5 B5)           */
/* ========================================================================== */

/**
 * Parameters for qwen_tts_synthesize().
 *
 * Mirrors the fields of QwenTTSParams (see qwen_tts.h) actually consumed by
 * QwenTTS::generate(). Mode is inferred from which fields are populated:
 *   mode="icl"         → ref_audio_path + ref_text
 *   mode="xvec"        → ref_audio_path (.wav auto-extracts to .xvec tempfile)
 *   mode="customvoice" → speaker (built-in voice id resolved via voices.json)
 *
 * Zero-defaulted sampling fields use the same defaults as the qwen_tts
 * CLI / Python reference. Set any field to its "use default" sentinel to
 * fall through.
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
    int         cp_groups;          /* reserved; ignored in v1.2 (no longer in sampling params) */
    int         cp_layers;          /* reserved; ignored in v1.2                */
    int         greedy;             /* 0 = sample, non-zero = greedy            */
} qwen_tts_synth_params_t;

/**
 * One-shot text-to-speech synthesis.
 *
 * Dispatches to QwenTTS::generate() with params populated from the mode
 * field. Library allocates the output PCM buffer via malloc() (size is
 * unknown up front); caller MUST release it with qwen_tts_pcm_free().
 *
 * pcm_out:       [out] pointer receiving the malloc()'d f32 buffer (24kHz mono)
 * n_samples_out: [out] number of float samples written
 *
 * Returns 0 on success. On any error, *pcm_out is set to NULL and
 * *n_samples_out is set to 0. Error codes:
 *   -1: ctx or params was NULL
 *   -2: unknown mode
 *   -3: generation failed
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
