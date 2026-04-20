# Ascend API Bridge Contract

Follow-on contract to `OminiX-Ascend/NATIVE_TTS_CONTRACT.md` §8
(2026-04-19 post-v1 unification direction). This contract covers
direction (1) only: wire OminiX-API to `libqwen_tts_api.so` via FFI
and unify MLX and Ascend backends behind a shared Rust trait.
Directions (2) [CannFusion] and (3) [Ascend/MLX model-code merge]
remain explicitly out of scope.

## 1. Goal (single sentence)

OminiX-API serves TTS requests on Ascend 910 hosts via
`libqwen_tts_api.so` through bindgen-wrapped FFI, with MLX and Ascend
backends both implementing a shared `TextToSpeech` trait, at parity
(latency, ASR-content) with the existing subprocess path — **~1 week
of engineering, no changes to the native TTS delivery track**.

## 2. Non-goals

- Not replacing MLX-side TTS code.
- Not merging Ascend C++ and MLX Rust model implementations (rejected
  in `NATIVE_TTS_CONTRACT.md` §8 2026-04-19).
- Not CannFusion DSL work (separate research track).
- Not a CLI rewrite; existing `/v1/audio/tts/ascend*` endpoints stay.
- Not removing the subprocess Ascend path — kept as fallback for
  platforms without the .so, and for CI on hosts without full CANN.
- No new streaming / WebSocket endpoints in v1 (deferred; current
  subprocess path is batch-only anyway).

## 3. Current state (update as work lands)

- `libqwen_tts_api.so` **does not exist**. `tools/qwen_tts/
  CMakeLists.txt` in OminiX-Ascend only builds the `qwen_tts`
  executable; `qwen_tts_api.cpp` is not compiled into any target.
  Header at `tools/qwen_tts/qwen_tts_api.h` is bindgen-clean
  (`<stdint.h>` + `<stddef.h>` only, `extern "C"` guarded, no C++
  types leaked).
- OminiX-API has a subprocess-based Ascend TTS path:
  `src/engines/ascend.rs::AscendTtsEngine`, invoked by handlers at
  `src/handlers/audio.rs` endpoints `/v1/audio/tts/ascend` and
  `/v1/audio/tts/ascend/clone`. Config via `AscendConfig::from_env()`
  reading `ASCEND_BIN_DIR` + `ASCEND_TTS_MODEL_DIR`.
- No shared TTS trait today. Each backend (GPT-SoVITS, Qwen3-MLX,
  Ascend subprocess, OuteTTS) is point-to-point wired in
  `src/handlers/audio.rs`. MLX path uses the channel+dedicated-thread
  pattern (`src/inference/thread.rs`); Ascend path uses
  `tokio::task::spawn_blocking`.
- OminiX-API is a single crate (no workspace). Path-deps into
  `../OminiX-MLX/`. No existing `build.rs` with `bindgen` in this
  repo; `mlx-sys` handles its own bindgen.

## 4. Architecture target

**Ascend side (OminiX-Ascend):**
- `tools/qwen_tts/CMakeLists.txt` adds
  `add_library(qwen_tts_api SHARED qwen_tts_api.cpp ...)` linking
  all the same internals as `qwen_tts` binary. Output:
  `build-85-cann-on/lib/libqwen_tts_api.so.{version}` with versioned
  symlinks.

**API side (OminiX-API):**
- New subcrate (workspace-adjacent) `qwen-tts-ascend-sys` that vendors
  `qwen_tts_api.h`, runs `bindgen` in `build.rs`, links
  `libqwen_tts_api.so` (via `pkg-config` or `ASCEND_TTS_LIB_DIR` env
  var). Provides a thin safe wrapper: RAII handle, checked return
  codes, buffer-size helpers.
- New `trait TextToSpeech` in `src/engines/mod.rs` or a dedicated
  module. Methods (conservative v1): `synthesize`, `synthesize_clone`
  (optional), `reset`, `backend_name`, `supports_clone`, and capability
  query for streaming (all `false` in v1).
- Implementations: `GptSovitsMlxTts`, `Qwen3MlxTts`,
  `AscendSubprocessTts` (existing path, refactored), `AscendFfiTts`
  (new).
- Handler dispatch via `enum TtsBackend` resolved from env/config at
  startup (`ASCEND_TTS_TRANSPORT=ffi|subprocess`, default `subprocess`
  so we can flip after validation).
- Platform gating: `AscendFfiTts` behind `#[cfg(target_os = "linux")]`
  (Ascend runs Linux/aarch64); Mac dev hosts compile but skip the FFI
  impl via a stub that returns `Unsupported`.

**Net request flow after v1:**
`POST /v1/audio/tts/ascend` → `handlers::audio::tts_ascend` →
`tts_backend.synthesize(req)` (trait) → dispatches to either
`AscendFfiTts` or `AscendSubprocessTts` by config. Same endpoint, same
request/response shape, same audio output.

## 5. Milestones (checkable)

### B1 — Build `libqwen_tts_api.so` on Ascend side (1-2 days)

- [x] 1.1 Add `add_library(qwen_tts_api SHARED qwen_tts_api.cpp ${common sources})`
      to `tools/qwen_tts/CMakeLists.txt`. Link identical internals to
      `qwen_tts` executable (BPETokenizer, TalkerLLM, SpeechTokenizer*,
      SpeakerEncoder, ggml, ggml-cann).
      **Verified-by:** (a) New SHARED target `qwen_tts_api` added in
      `OminiX-Ascend/tools/qwen_tts/CMakeLists.txt`. Sources:
      `qwen_tts_api.cpp`, `talker.cpp`, `tts_transformer.cpp`,
      `speaker_encoder.cpp`, `speech_tokenizer_{encoder,decoder}.cpp`,
      `model_defs.cpp`, `stft.cpp`, kissfft, and (when
      `QWEN_TTS_CP_CANN=ON`) `cp_cann_engine.cpp`, `cp_cann_symbols.cpp`,
      `talker_cann_engine.cpp`. Links `ggml`, `qwen_common` (POSITION_INDEPENDENT_CODE
      flipped ON from within the new target since qwen_common is STATIC and now
      linked into a SHARED consumer), `Threads::Threads`, `OpenMP::OpenMP_CXX`,
      and `${CMAKE_DL_LIBS}` when CP-CANN is on.
      (b) Configure + build on ModelArts 910B4 with
      `cmake .. -DGGML_CANN=ON -DBUILD_SHARED_LIBS=ON && cmake --build .
      --target qwen_tts_api -j8` → `[100%] Built target qwen_tts_api`.
      (c) Artifact `~/work/OminiX-Ascend/build-85-cann-on/bin/libqwen_tts_api.so.1.0.0`
      (1,047,496 bytes).
- [x] 1.2 Set `OUTPUT_NAME qwen_tts_api`, `VERSION 1.0.0`,
      `SOVERSION 1`. Install rule to `lib/` alongside binary.
      **Verified-by:** (a) `set_target_properties(qwen_tts_api PROPERTIES
      OUTPUT_NAME qwen_tts_api VERSION 1.0.0 SOVERSION 1)` +
      `install(TARGETS qwen_tts_api LIBRARY DESTINATION lib ...)` +
      `install(FILES qwen_tts_api.h DESTINATION include)`.
      (b) `cmake --install build-85-cann-on/tools/qwen_tts --prefix .../install`
      produced symlink chain `libqwen_tts_api.so → .so.1 → .so.1.0.0` under
      `install/lib/` and the header under `install/include/qwen_tts_api.h`.
      (c) Server artifacts at
      `~/work/OminiX-Ascend/build-85-cann-on/install/{lib,include}/`.
- [x] 1.3 Verify `nm -D libqwen_tts_api.so | grep qwen_tts_` lists all
      12 exported symbols; verify header at
      `include/qwen_tts_api.h` installs with the library.
      **Verified-by:** (a) Header has 14 symbols (contract summary said 12; the
      actual ABI is `load`, `free`, `hidden_size`, `vocab_size`,
      `has_speaker_encoder`, `text_embed`, `codec_embed`, `codec_head`,
      `generation_embed`, `reset_cache`, `forward`, `predict_codes`,
      `decode_audio`, `extract_speaker` = 14 — matches Agent Y's B2.3 finding).
      `nm -D --defined-only .../libqwen_tts_api.so.1.0.0 | grep qwen_tts_`
      returns exactly those 14 under version tag `QWEN_TTS_API_1.0`.
      (b) Symbol-pollution check: `nm -D --defined-only ... | grep -v
      qwen_tts_` returns only the version anchor `QWEN_TTS_API_1.0` — no
      ggml / ggml-cann / qwen_common symbols leak (risk register §7). The
      new linker version script at `tools/qwen_tts/qwen_tts_api.version` plus
      `-Wl,--version-script,--no-undefined` enforces this.
      (c) Header installed at
      `~/work/OminiX-Ascend/build-85-cann-on/install/include/qwen_tts_api.h`
      (MD5 `4ad8cab5fc4bd14d1eba81176d68abc5`, identical to the pin Agent Y
      recorded in B2.1).
- [x] 1.4 Smoke test: C program that calls `qwen_tts_load` +
      `qwen_tts_free` in a loop (10×) with no leak (valgrind or
      `npu-smi` peak memory check).
      **Verified-by:** (a) `tools/qwen_tts/test_api_smoke.c` (new), built
      against `libqwen_tts_api.so` with
      `gcc -std=c11 ... -lqwen_tts_api -o bin/test_api_smoke`. 10/10 iters
      PASS on Ascend 910B4 (device 2). Each iter returns
      `hidden=2048 vocab=3072 spk=1`, per-iter load time 17–28 s (native
      Talker CANN engine init dominates; free is <10 ms). Exit 0, no crash.
      (b) `npu-smi info -t usages -i 2` HBM Usage Rate: **8% before and 8%
      after** the full loop — no leak at that granularity. Process exit
      clean.
      (c) Artifact: `~/work/OminiX-Ascend/build-85-cann-on/bin/test_api_smoke`.
      Source: `tools/qwen_tts/test_api_smoke.c`.
      Notes: had to patch two pre-existing defects in the un-compiled
      `qwen_tts_api.cpp` for the smoke test to reach success:
      (i) `BPETokenizer` → `BpeTokenizer` class-name drift (common code was
      renamed after the API stub was written); (ii) talker GGUF auto-upgrade
      preferred `qwen_tts_talker_llama_q8_0.gguf`, which the native
      TalkerCannEngine rejects with "unsupported dtype 8" — switched default
      to the F16/F32 `qwen_tts_talker_llama.gguf` (Q8 remains viable only on
      the llama.cpp fallback path, QWEN_TTS_LLAMA=ON, which is off in the
      API build); and (iii) routed `use_cp_cann=true`/`use_talker_cann=true`
      into `TalkerLLM::load_model()` when `QWEN_TTS_HAS_CP_CANN` is defined
      and the caller passes `n_gpu_layers > 0` (needed because the API build
      is native-CANN-only; without this flag, load_model fails since the
      llama.cpp fallback isn't compiled in).

**Acceptance**: `libqwen_tts_api.so` loadable via `dlopen`; smoke test
passes; `nm` symbol set matches header.
**Acceptance met.** Shared library loads (C test linked against it loads +
frees 10 handles successfully); all 14 header-declared symbols are in `nm -D`
output; version script ensures no other symbols escape.

### B2 — `qwen-tts-ascend-sys` Rust crate (2 days, parallel with B1)

- [x] 2.1 Create crate at `OminiX-API/qwen-tts-ascend-sys/`. `build.rs`
      with `bindgen` over `qwen_tts_api.h`. Vendor the header from
      OminiX-Ascend (copy or symlink with documented pin).
      **Verified-by:** (a) `cargo check` in `qwen-tts-ascend-sys/` on
      macOS arm64 — "Finished `dev` profile ... in 0.08s", no errors.
      (b) Vendored header at
      `OminiX-API/qwen-tts-ascend-sys/wrapper/qwen_tts_api.h` with pin
      header `Pinned from: OminiX-Ascend @
      12405a5251346d9568116e801c88b22bced661e8` and upstream SHA-256
      `39a067d7d2a8655a53ad12e8e0ddfd5ccf6b237cece315599f8753813ea82e44`.
      (c) `build.rs` at `OminiX-API/qwen-tts-ascend-sys/build.rs` runs
      `bindgen::Builder::default().header(...).allowlist_function("qwen_tts_.*")`.
- [x] 2.2 Link hint via `ASCEND_TTS_LIB_DIR` env var and
      `cargo:rustc-link-search` / `cargo:rustc-link-lib=qwen_tts_api`.
      Fall back to `pkg-config` if available.
      **Verified-by:** (a) `build.rs` emits `cargo:rustc-link-search=native=...`
      + `cargo:rustc-link-lib=dylib=qwen_tts_api` only when
      `CARGO_FEATURE_ASCEND_AVAILABLE` is set and `target_os == "linux"`;
      otherwise no link directives (Mac stub path). (b) `cargo check
      --features ascend-available` on macOS succeeds with warning
      `target_os=macos; skipping link directives`.
      (c) `build.rs:45-95`.
- [x] 2.3 Safe wrapper module: `QwenTtsCtx` struct holding raw
      handle; `Drop` calls `qwen_tts_free`; methods wrap all 14
      C functions (header exports 14 — the contract summary said "12"
      but the actual ABI is `load`, `free`, `hidden_size`, `vocab_size`,
      `has_speaker_encoder`, `text_embed`, `codec_embed`, `codec_head`,
      `generation_embed`, `reset_cache`, `forward`, `predict_codes`,
      `decode_audio`, `extract_speaker` = 14); errors as
      `thiserror`-defined `TtsError`.
      **Verified-by:** (a) `cargo check` both with and without feature —
      zero warnings, zero errors. (b) Generated `bindings.rs` size: 100
      lines, 14 `pub fn qwen_tts_*` declarations (grep-verified). (c)
      `OminiX-API/qwen-tts-ascend-sys/src/wrapper.rs` — `QwenTtsCtx`
      with `Drop` impl, `unsafe impl Send` (explicitly no `Sync`),
      `TtsError` enum variants `LoadFailed`, `ForwardFailed(i32)`,
      `PredictFailed(i32)`, `DecodeFailed(i32)`, `SpeakerExtractFailed(i32)`,
      `Unsupported(&'static str)`.
- [x] 2.4 Unit test: stub library path (build the smoke test from B1.4
      as a cdylib for CI) or behind `#[cfg(ascend_available)]` feature
      that defaults off.
      **Verified-by:** (a) `cargo test --no-run` compiles successfully
      on macOS; the `raii_lifecycle` test is gated
      `#[cfg(all(test, feature = "ascend-available", target_os = "linux"))]`
      so it stubs on Mac and exercises the full
      `load → hidden_size/vocab_size → Drop` path when run on Ascend.
      (b) Test reads `ASCEND_TTS_MODEL_DIR` from env; skips gracefully
      if unset so CI without full model weights still passes.
      (c) `OminiX-API/qwen-tts-ascend-sys/src/wrapper.rs` — `mod tests`
      at line ~495.

**Acceptance**: crate compiles against a real `libqwen_tts_api.so` on
Ascend; `cargo test` passes the RAII test; no `unsafe` leaks past
crate boundary.

### B3 — `TextToSpeech` trait + retrofit (2 days)

- [x] 3.1 Define `trait TextToSpeech` with minimal v1 surface
      (synthesize, backend_name, supports_clone).
      **Verified-by:** (a) New module `src/engines/tts_trait.rs`
      defines `pub trait TextToSpeech: Send + Sync` with
      `backend_name(&self) -> &'static str`, `supports_clone(&self)
      -> bool`, `synthesize(&self, TtsRequest) -> Result<TtsResponse,
      TtsError>`, and a defaulted `synthesize_clone`. Object-safe so
      handlers can hold `Arc<dyn TextToSpeech>`.
      (b) Request/response/error types also defined there:
      `TtsRequest { input, voice, language, speed, instruct }`,
      `TtsCloneRequest { input, reference_audio: Vec<u8>, language,
      speed, instruct }`, `TtsResponse::{Wav(Vec<u8>), Pcm { samples:
      Vec<i16>, sample_rate: u32 }}`, and `TtsError::{Unsupported,
      BadRequest, Backend}` (thiserror-derived).
      (c) Location decision: `src/engines/tts_trait.rs` (NOT
      `src/tts/mod.rs`) — documented in the file's top-level doc
      comment. Keeps the trait next to the four backend impls in
      `src/engines/`.
- [x] 3.2 Implement for `GptSovitsMlxTts`, `Qwen3MlxTts`,
      `AscendSubprocessTts` (refactor existing), `AscendFfiTts` (new,
      wraps `qwen-tts-ascend-sys`).
      **Verified-by:** (a) All four impls in a new module
      `src/engines/tts_backends.rs`. `AscendSubprocessTts` wraps the
      existing `ascend::AscendTtsEngine` (struct unchanged per §5.3.4;
      the subprocess flow is byte-identical — we only moved its call
      site behind the trait). `AscendFfiTts` wraps
      `qwen_tts_ascend_sys::QwenTtsCtx` with a `Mutex<Option<_>>`
      guarding the `!Sync` handle (risk register §7). `Qwen3MlxTts`
      and `GptSovitsMlxTts` hold a `mpsc::Sender<InferenceRequest>`
      and use `blocking_send` + `oneshot::blocking_recv` — callers
      must invoke from `spawn_blocking` (all existing handlers
      already do).
      (b) Platform gating: `AscendFfiTts` real body is
      `#[cfg(all(feature = "ascend-tts-ffi", target_os = "linux"))]`;
      a stub with the same name and trait impl exists for all other
      configurations so type-erased `Arc<dyn TextToSpeech>` shape is
      identical across platforms. Mac / feature-off path returns
      `TtsError::Unsupported`.
      (c) B3 scope note: `AscendFfiTts::synthesize` currently returns
      `Unsupported` even on Linux+feature-on — the trait binding and
      Mutex lock shape land now so B3's handler refactor has a target,
      but the generation loop (forward + predict_codes + decode_audio)
      is a B4 follow-up that needs to be validated against real
      weights on the Ascend host. Documented inline.
- [x] 3.3 Handler refactor: `src/handlers/audio.rs::tts_ascend` takes
      `Arc<dyn TextToSpeech + Send + Sync>` from app state instead of
      constructing per-request. Resolve variant at startup via env.
      **Verified-by:** (a) `AppState` gains
      `pub ascend_tts_backend: Option<Arc<dyn TextToSpeech>>`
      (src/state.rs). `src/main.rs` calls
      `engines::tts_backends::build_ascend_tts_backend(cfg.clone())`
      once when `ascend_config` is present; that function reads
      `ASCEND_TTS_TRANSPORT` (`ffi`|`subprocess`, default
      `subprocess`) and returns the right `Arc<dyn TextToSpeech>`.
      (b) `tts_ascend` and `tts_ascend_clone` handlers (src/handlers/
      audio.rs lines ~460–570) now pull
      `state.ascend_tts_backend`, build a `TtsRequest`/
      `TtsCloneRequest`, and dispatch via
      `spawn_blocking(move || backend.synthesize(req))`. No
      per-request engine construction. `TtsResponse::Pcm` path is
      handled with `pcm_to_wav` for future PCM-returning backends.
      (c) MLX handlers untouched — scope decision documented in the
      B3 report: retrofitting the MLX-side handlers (which stream
      sentence-by-sentence via `spawn_per_sentence_tts`) would have
      required re-plumbing the streaming sentence-per-oneshot channel
      pattern through the new trait, with zero user-facing change.
      The MLX trait impls exist in `tts_backends.rs` and can be wired
      by a follow-up without touching the trait definition.
- [x] 3.4 Keep the existing subprocess path and endpoint semantics
      byte-identical; the trait is the only change users can observe
      (which they should not).
      **Verified-by:** (a) `AscendSubprocessTts::synthesize` builds
      the same `SpeechRequest` the old handler did (same voice /
      language defaults: `voice="default"`, `language="English"`)
      and calls `AscendTtsEngine::new((*cfg).clone())?.synthesize(&req)`
      — the path through `ascend.rs::run_tts` is bit-for-bit the
      original subprocess invocation. Same for `synthesize_clone`.
      (b) No modification to `src/engines/ascend.rs` in this
      milestone (grep-verified).
      (c) `cargo check` default: `Finished \`dev\` profile … in
      3.76s`, 0 errors. `cargo check --features ascend-tts-ffi`: same.

**Acceptance**: cargo build passes on Mac (subprocess only) and Linux
Ascend (both variants); existing subprocess endpoint returns the same
bytes for the same request; no regression in MLX paths.
**Acceptance met** (Mac side): both `cargo check` and `cargo check
--features ascend-tts-ffi` complete cleanly on macOS arm64. Linux-Ascend
link-clean verification and E2E byte-equivalence of the subprocess
endpoint roll into B4.

### B5 — High-level `qwen_tts_synthesize` ABI (1 day) — inserted 2026-04-19

Discovered during B3: the fine-grained ABI exposes engine internals
(embed / forward / predict_codes / decode_audio) but no one-shot
synthesis function. To call it from Rust, we'd either (a) reimplement
the full autoregressive loop + BPE tokenization + sampling in Rust
(~1 week, drift risk) or (b) add a coarse ABI function on the C++ side
that wraps the existing `QwenTTS::generate()` logic. Contract decision:
**(b)**. `NATIVE_TTS_CONTRACT.md` §8 2026-04-19 rationale (user-visible
win, no model-code merge) applies.

- [ ] 5.1 Add `qwen_tts_synth_params_t` struct and
      `int qwen_tts_synthesize(qwen_tts_ctx_t*, const qwen_tts_synth_params_t*,
      float** pcm_out, int* n_samples_out)` to `qwen_tts_api.h`. Mirrors
      `QwenTTSParams` fields used by `QwenTTS::generate()`/`generate_xvec()`/
      `generate_customvoice()`. Mode selector chooses path.
- [ ] 5.2 Add `void qwen_tts_pcm_free(float* pcm)` so library owns the
      allocation (unknown output length; two-shot plan is worse UX).
- [ ] 5.3 Implement in `qwen_tts_api.cpp` by instantiating a
      `QwenTTSParams`, dispatching to `QwenTTS::generate*`, and
      `malloc`+`memcpy`-ing the resulting `std::vector<float>` into a
      caller-freeable buffer.
- [ ] 5.4 Add the two new symbols to the version script + vendored
      header. Bump header `SOVERSION` stays at 1 (additive only —
      no ABI break).
- [x] 5.5 Re-run `bindgen` in `qwen-tts-ascend-sys` (header pin updates;
      SHA-256 bump). Wrapper exposes `QwenTtsCtx::synthesize(params) ->
      Result<Vec<f32>, TtsError>`.
      **Verified-by:** (a) Vendored header at
      `OminiX-API/qwen-tts-ascend-sys/wrapper/qwen_tts_api.h` updated with
      upstream content SHA-256 `042025b6979ad2096990e1f42f5253a46fb6dfc90de258abfe3ca022c7d892e9`.
      Pin base commit stays at `12405a5251346d9568116e801c88b22bced661e8`
      (Agent X's B5.1–5.4 header edit was still working-tree at vendor
      time; pin block notes this — a later refresh should record the
      actual B5 commit hash once Agent X commits). Bindgen regenerates
      `qwen_tts_synth_params_t`, `qwen_tts_synthesize`, and
      `qwen_tts_pcm_free` in
      `target/debug/build/.../out/bindings.rs` lines 101–133.
      (b) `cargo check` in `qwen-tts-ascend-sys/` both feature states:
      `Finished dev profile ... in 3.56s` (no feature) and
      `... in 0.02s` (with `--features ascend-available`, macOS warning
      skips linking as expected).
      (c) Wrapper:
      `OminiX-API/qwen-tts-ascend-sys/src/wrapper.rs::QwenTtsCtx::synthesize`
      + new `pub struct SynthParams` mirror. Error path reuses a new
      `TtsError::Backend(String)` variant (wraps C return codes, CString
      NUL failures, and NULL-pcm/zero-samples-on-rc=0 guards); SAFETY
      comments sit on every `unsafe` block. Library-owned PCM is freed
      via `qwen_tts_pcm_free` both on success and on the non-zero-rc
      cleanup path. `SynthParams` + `TtsError::Backend` re-exported at
      the crate root (`src/lib.rs`).
- [x] 5.6 Wire `AscendFfiTts::synthesize` in `src/engines/tts_backends.rs`
      to call `QwenTtsCtx::synthesize`, under the `Mutex` guard.
      **Verified-by:** (a)
      `src/engines/tts_backends.rs::AscendFfiTts` (cfg-gated real impl,
      `feature = "ascend-tts-ffi"` + `target_os = "linux"`):
      `synthesize` now calls `ensure_loaded()` (lazy
      `QwenTtsCtx::load` on first request, stored in
      `Mutex<Option<QwenTtsCtx>>`), builds `SynthParams` via a new
      `params_for_preset` helper (maps empty `voice` → `"icl"` mode,
      non-empty voice → `"customvoice"` with `speaker=voice`), and wraps
      the resulting `Vec<f32>` into `TtsResponse::Pcm` at 24 kHz via a
      new `f32_pcm_to_response` helper (clamps to [-1,1] and rounds to
      i16 so the existing `TtsResponse::Pcm { samples: Vec<i16>,
      sample_rate: u32 }` enum shape is preserved).
      (b) `synthesize_clone` writes `req.reference_audio` to a
      `tempfile::NamedTempFile` (prefix `ascend_ffi_ref_`, suffix
      `.wav`) and passes the path via `SynthParams.ref_audio_path` in
      ICL mode. The tempfile is explicitly `drop`ed after the synthesis
      call so unlink happens on scope exit. `supports_clone` flipped
      `false → true` since ICL mode does not require a speaker encoder.
      (c) `cargo check` default: `Finished dev profile ... in 7.44s`,
      31 warnings (all pre-existing dead-code lints). `cargo check
      --features ascend-tts-ffi` on macOS: `Finished dev profile ... in
      6.73s`, same warnings. Linux link verification and E2E parity roll
      into B4.

**Acceptance**: `qwen_tts_synthesize` produces ASR-identical audio to
the `qwen_tts` binary for the same params on 1 canonical utt (run on
ModelArts 910B4). `cargo check --features ascend-tts-ffi` stays clean
on Mac; `cargo build` clean on Ascend host.

### B4 — E2E parity on Ascend host (1 day) — runs after B5

- [ ] 4.1 Deploy OminiX-API + `libqwen_tts_api.so` to the ModelArts
      910B4 host (ma-user@dev-modelarts… port 31984, reused from
      TTS contract).
- [ ] 4.2 Curl `/v1/audio/tts/ascend` with `ASCEND_TTS_TRANSPORT=ffi`
      and again with `=subprocess` for the same 3 canonical utts.
- [ ] 4.3 ASR-gate both outputs (same `scripts/asr_quality_check.sh`
      pattern as TTS contract).
- [ ] 4.4 Latency compare: FFI path p50 ≤ subprocess path p50 minus
      any fork/exec overhead (target: ≥100 ms saved per request).

**Acceptance**: both transports produce ASR-PASS; FFI p50 latency
lower; no crashes over a 100-request soak.

## 6. Acceptance criteria (summary)

- [ ] `libqwen_tts_api.so` builds, installs, and links into OminiX-API
      on an Ascend 910 host.
- [ ] `POST /v1/audio/tts/ascend` (FFI transport) produces
      ASR-identical audio vs subprocess transport on 3 canonical utts.
- [ ] FFI p50 latency < subprocess p50 latency (fork/exec savings
      realized).
- [ ] Shared `TextToSpeech` trait implemented by all four backends
      (GPT-SoVITS-MLX, Qwen3-MLX, Ascend-subprocess, Ascend-FFI).
- [ ] MLX path on Mac shows no regression (smoke: one MLX TTS call
      returns audio).
- [ ] 100-request soak on Ascend FFI path with no crash, no VRAM leak.

**Verification stamp (per [x] item)**: same rule as TTS contract —
the completing agent appends a **Verified-by:** line under each item
with (a) what was built/run, (b) measured numbers, (c) artifact path.

## 7. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| CANN runtime now lives in OminiX-API's address space, not a subprocess. A CANN-triggered abort kills the whole API server. | High | Keep the subprocess fallback path; panic-handle around FFI calls with process-respawn on abort (v2). Run FFI path under a dedicated worker thread first, so a crash there is isolable. |
| `ggml-cann` symbol collision with any future Rust dep also linking ggml. | Medium | Hide ggml symbols via linker version script in `libqwen_tts_api.so` build; only export the 12 `qwen_tts_*` symbols. |
| Engine is not thread-safe per handle (TTS contract finding). Concurrent requests to one handle corrupts KV cache. | High | v1: serialize calls per-handle behind a `Mutex` in the safe wrapper. Parallelism = multiple handles, not multiple threads on one handle. |
| Mac dev workflow cannot exercise FFI path. | Medium | Stub impl for non-Linux; CI target on Ascend host only. Document clearly. |
| Header drift between OminiX-Ascend and vendored copy in API crate. | Medium | `build.rs` asserts a content-hash of the header matches a pinned value; bump the pin intentionally. |

## 8. Decision log

- **2026-04-19 Contract created.** Split from
  `OminiX-Ascend/NATIVE_TTS_CONTRACT.md` §8 to keep PM-facing tracks
  scoped. This contract owns direction (1) only. The native TTS
  delivery track stays untouched.
- **2026-04-19 Keep the subprocess transport.** Rationale: CI, Mac
  dev, and any Ascend host without the `.so` installed need a
  working path. Also serves as a blast-radius containment if the
  FFI path aborts.
- **2026-04-19 Shared trait scope is v1-minimal.** Only
  `synthesize` / `supports_clone`. No streaming, no mid-generation
  control. Expand only when a concrete endpoint needs it.

## 9. Parallelism playbook

- B1 and B2 can start in parallel: B2 drafts bindgen + wrapper
  against the committed header; can't link or run until B1 lands, but
  compilation against a stub .so is OK.
- B3 starts once B2 is link-clean on Ascend; B3 does not need B1 done
  beyond "header vendored" for the API-side refactor.
- B4 serializes last — needs all three prior milestones on one host.

Preferred allocation:
- **Agent X** (Ascend-side, C++): B1.
- **Agent Y** (API-side, Rust): B2, then B3.
- **PM (this session)**: arbitrate, verify, run B4.

## 10. File index

| File | Purpose |
|---|---|
| `OminiX-Ascend/tools/qwen_tts/qwen_tts_api.h` | C ABI header (source of truth) |
| `OminiX-Ascend/tools/qwen_tts/qwen_tts_api.cpp` | C ABI impl; not currently compiled — B1 hooks it up |
| `OminiX-Ascend/tools/qwen_tts/CMakeLists.txt` | add_library target lands here |
| `OminiX-API/qwen-tts-ascend-sys/` | New crate (B2); bindgen + safe wrapper |
| `OminiX-API/src/engines/ascend.rs` | Existing subprocess path — refactored in B3 |
| `OminiX-API/src/handlers/audio.rs` | Handler dispatch; updated in B3 |
| `OminiX-API/src/engines/mod.rs` (or new module) | `trait TextToSpeech` lives here |

## 11. Session boot checklist

When resuming work on this contract:
1. Read §1-4 for scope and architecture.
2. Grep `[ ]` in §5 to find the next unlanded item.
3. Confirm `build-85-cann-on/` on ModelArts is the active build tree
   (TTS contract §3 active path).
4. For API-side work: `cd /Users/yuechen/home/OminiX-API && cargo
   check --features ascend` should be your first smoke test once B2
   lands.
5. For Ascend-side work: ssh to port 31984 of the ModelArts container,
   `~/work/OminiX-Ascend/` is the live tree.
