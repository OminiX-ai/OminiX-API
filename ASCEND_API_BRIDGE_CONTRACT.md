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

- [ ] 1.1 Add `add_library(qwen_tts_api SHARED qwen_tts_api.cpp ${common sources})`
      to `tools/qwen_tts/CMakeLists.txt`. Link identical internals to
      `qwen_tts` executable (BPETokenizer, TalkerLLM, SpeechTokenizer*,
      SpeakerEncoder, ggml, ggml-cann).
- [ ] 1.2 Set `OUTPUT_NAME qwen_tts_api`, `VERSION 1.0.0`,
      `SOVERSION 1`. Install rule to `lib/` alongside binary.
- [ ] 1.3 Verify `nm -D libqwen_tts_api.so | grep qwen_tts_` lists all
      12 exported symbols; verify header at
      `include/qwen_tts_api.h` installs with the library.
- [ ] 1.4 Smoke test: C program that calls `qwen_tts_load` +
      `qwen_tts_free` in a loop (10×) with no leak (valgrind or
      `npu-smi` peak memory check).

**Acceptance**: `libqwen_tts_api.so` loadable via `dlopen`; smoke test
passes; `nm` symbol set matches header.

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

- [ ] 3.1 Define `trait TextToSpeech` with minimal v1 surface
      (synthesize, backend_name, supports_clone).
- [ ] 3.2 Implement for `GptSovitsMlxTts`, `Qwen3MlxTts`,
      `AscendSubprocessTts` (refactor existing), `AscendFfiTts` (new,
      wraps `qwen-tts-ascend-sys`).
- [ ] 3.3 Handler refactor: `src/handlers/audio.rs::tts_ascend` takes
      `Arc<dyn TextToSpeech + Send + Sync>` from app state instead of
      constructing per-request. Resolve variant at startup via env.
- [ ] 3.4 Keep the existing subprocess path and endpoint semantics
      byte-identical; the trait is the only change users can observe
      (which they should not).

**Acceptance**: cargo build passes on Mac (subprocess only) and Linux
Ascend (both variants); existing subprocess endpoint returns the same
bytes for the same request; no regression in MLX paths.

### B4 — E2E parity on Ascend host (1 day)

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
