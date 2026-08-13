# ASR batching modes

Qwen3-ASR requests can be coalesced so several utterances share one decode pass.
This trades latency for throughput, and `ASR_MODE` picks where on that trade-off
the server sits.

```bash
ASR_MODE=conversational ominix-api          # default
ominix-api --asr-mode interactive
ASR_MODE=offline ASR_MAX_BATCH=48 ominix-api
```

| Mode | Max batch | Wait window | Use when |
|------|-----------|-------------|----------|
| `off` | 1 | — | Debugging, or reproducing pre-batching behaviour |
| `interactive` | 4 | 10 ms | Latency budget under ~300 ms — voice UI, command-and-control |
| `conversational` *(default)* | 8 | 25 ms | Live captioning, dictation, meeting transcription |
| `offline` | 32 | 150 ms | Bulk transcription, where only throughput matters |

## Measured

Apple M5 Max (40 GPU cores, 128 GB), Qwen3-ASR-1.7B 4-bit, MLX 0.32.0, driven by
LibriSpeech `test-clean`. Throughput and latency at 16 concurrent clients:

| Mode | Throughput | p50 | p95 | vs `off` |
|------|-----------|-----|-----|----------|
| `off` | 38.0 ×RT | 2,391 ms | 4,016 ms | 1.00× |
| `interactive` | 66.8 ×RT | 1,270 ms | 2,571 ms | 1.76× |
| `conversational` | 73.2 ×RT | 1,249 ms | 2,470 ms | **1.93×** |
| `offline` | 79.8 ×RT | 981 ms | 2,229 ms | 2.10× |

Batching improves latency here as well as throughput: at a fixed number of
waiting clients the queue drains almost twice as fast, so each request spends
less time waiting even though its own decode step is shared.

Sustained real-time sessions (open-loop, arrivals paced by wall clock):

| Sessions | p50 | Verdict |
|----------|-----|---------|
| 60 | 358 ms | OK |
| **70** | **487 ms** | **OK — sustainable ceiling** |
| 80 | 2,111 ms | saturated |

That is **70 sessions against 40 unbatched**, a 1.75× improvement from the mode
switch alone.

Correctness is unchanged: across 48 utterances, transcripts produced in batches
of 8 were **identical to those produced one at a time**, with the same 2.47% WER.

`ASR_MAX_BATCH` / `--asr-max-batch` overrides the batch size without changing the
wait window. Aliases are accepted: `low-latency` → `interactive`,
`balanced`/`default` → `conversational`, `batch`/`throughput` → `offline`.

## Requires MLX 0.32.0 or newer

**Batching does nothing on an older MLX, by design.** MLX before 0.32.0 computes
RoPE incorrectly when the sequence length is 1 and the batch is larger than 1 —
rows after the first come back wrong, often all zeros. That is the exact shape a
batched decode step uses every token, so on an affected build batching would
return fluent but wrong transcripts for every request except the first in each
batch. Silent corruption, not an error.

Measured on an M5 Max with `mx.fast.rope` over a `(2, 4, 1, 8)` input of ones,
comparing the two identical rows:

| MLX | Result |
|-----|--------|
| 0.30.1 *(pinned by this workspace)* | broken |
| 0.30.3, 0.30.6 | broken |
| 0.31.0, 0.31.1, 0.31.2 | broken |
| 0.32.0 | correct |

`qwen3_asr_mlx::batched_decode_supported()` probes for this at startup — running
the actual defect rather than comparing version strings, so a patched or vendored
build is judged on behaviour. When it reports a bad build, `AsrEngine` refuses to
batch regardless of `ASR_MODE`, logs a warning, and serves one request at a time.
Output stays correct; you simply do not get the speedup until MLX is bumped.

### Producing MLX 0.32.0 artifacts without full Xcode

Building MLX from source needs `xcrun metal`, which ships with full Xcode rather
than the Command Line Tools. The metallib is a standalone runtime file, though,
not something linked into `libmlx` — so the official `mlx-metal` wheel supplies
the one piece that requires the Metal compiler, already built:

1. `pip install mlx==0.32.0` and take `mlx/lib/{libmlx.dylib, libjaccl.dylib,
   mlx.metallib}` from the wheel.
2. Compile the vendored `mlx-c` sources against MLX 0.32.0 headers and archive
   them into `libmlxc.a`. Four call sites need updating for MLX 0.32 API
   changes — `fft.cpp` (every transform gained an `FFTNorm` parameter),
   `ops.cpp` (`quantize`/`dequantize` gained `global_scale`, `qqmm` gained two),
   and `metal.cpp` (`metal::device_info()` became `device_info(Device)`).
   mlx-c's own C API is unchanged, so the generated Rust bindings stay valid.
3. Point `MLX_PREBUILT_PATH` at a directory holding those files. `build.rs`
   detects `libmlx.dylib` and links dynamically, staging the dylibs beside the
   executable next to the metallib.
4. Build with `RUSTFLAGS="-C link-arg=-Wl,-rpath,@loader_path"`. Cargo ignores
   `rustc-link-arg` from a dependency's build script, so the rpath has to come
   from the top-level build.

`mlx-prebuilt-v0.1.0` cannot be used for this: besides being MLX 0.30.1, its
`libmlxc.a` was built without GGUF support, so `mlx_load_gguf` is undefined at
link time, and the archive ships no `mlx.metallib` in the API release tarball.

## Why batching helps

At batch 1, decode is limited by memory bandwidth rather than compute: each step
streams the whole weight matrix out of memory to produce a single token. Extra
sequences ride along on reads that were happening anyway, so step time grows far
more slowly than the work done. Measured on an M5 Max with the 4-bit 1.7B model,
64 sequences cost 6.8× the time of one — a 9.4× throughput gain — while the
encoder, which is compute-bound, was already running at 60% of peak GEMM
throughput and had nothing left to give.

## Why the modes look like this

Measured latency and capacity per batch size, calibrated against the live
server's 177.8 ms mean request:

| Batch | Latency | Sessions | Share of a 6.7 s utterance |
|-------|---------|----------|----------------------------|
| 1 | 178 ms | 40 | 2.6% |
| 4 | 281 ms | 94 | 4.2% |
| 8 | 459 ms | 110 | 6.8% |
| 16 | 807 ms | 121 | 12.0% |
| 32 | 1,228 ms | 156 | 18.3% |
| 64 | 2,438 ms | 155 | 36.3% |

Capacity plateaus after 32 while latency keeps doubling, so **64 is strictly
worse than 32** and no mode selects it. `ASR_MAX_BATCH` will let you go there;
there is no reason to.

## The batch forms itself

Coalescing is opportunistic. The inference thread takes whatever transcription
requests have already arrived and only lingers once a batch is already forming,
so a single request at low load is never held back — which is the usual reason
naive batching hurts latency.

No tuning is needed to land on the right batch size. Queue depth settles at
arrival rate times service time, so the batch that forms is the one the offered
load justifies: at 110 real-time sessions that is 16.4 requests/s against 459 ms
of service, which is 7.5 in flight — batch 8. The same identity holds at every
row of the table above.

## Operational notes

- **Only Qwen3-ASR batches.** Paraformer and SenseVoice+Qwen3-4B fall back to
  one request at a time regardless of mode; `AsrEngine::supports_batching()`
  reports which.
- **Mixed languages don't share a batch.** The language is baked into the decoder
  prompt, so requests whose `language` differs from the batch leader are served
  individually. Sending one language per client keeps batches full.
- **The 1024-token generation cap is load-bearing.** A runaway generation
  multiplies its KV cache by the batch size: at the crate default of 8192 tokens,
  batch 64 would reserve roughly 56 GB. Admission-control on projected KV rather
  than on request count if you raise it.
- **Failures are per request.** A bad audio file fails only its own caller; the
  rest of the batch is unaffected.
- **Watch MLX, not RSS.** Unified-memory allocations do not appear in process
  RSS — a worker holding 1.75 GB of KV cache reported 0.04 GB to `ps`. Use MLX's
  own memory counters.
