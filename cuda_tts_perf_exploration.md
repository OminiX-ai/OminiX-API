# Qwen3-TTS CUDA / GB10 — Performance Headroom Survey

**Profile date:** 2026-04-29  
**Host:** `zgx-3675` (NVIDIA GB10, sm_121a / Blackwell, 48 SMs, 24 MiB L2, **546.1 GB/s** unified LPDDR5X, 119 GiB)  
**Daemon:** `/home/user1/ominix-cuda/build-phase21/bin/tts_server` (Phase 2.10 warm)  
**Test prompt:** `"Hello world, this is the ominix CUDA TTS."` → 10.24 s audio  
**Warm wall-clock:** 6.66 – 7.21 s (RTF 0.65 – 0.70)  
**Profile:** `nsys profile -t cuda,nvtx,cublas,cudnn --cuda-graph-trace=node --delay=12 --duration=15`  
**Capture file:** `/tmp/tts_prof/tts_warm_113139.nsys-rep` on host  
**Source files referenced (all on host `zgx-3675`):**  
`/home/user1/ominix-cuda/tools/qwen_tts/native/talker_cuda_engine.cpp` (engine + Phase 2.5 graph capture, Phase 2.6 FP8 LM-head, attention),  
`/home/user1/ominix-cuda/tools/qwen_tts/native/tts_server.cpp` (Phase 2.10 daemon),  
`/home/user1/ominix-cuda/tools/qwen_tts/native/cuda_kernels/{attn_gqa.cu,rmsnorm.cu,rope_neox.cu,swiglu.cu,decoder_ops.cu,elementwise.cu}`.

---

## 1. nsys Top-10 Kernels (warm window, ~1.7 requests captured before duration cutoff)

| Rank | % GPU | Total (ns) | Calls | Avg (us) | Kernel |
|------|------:|-----------:|------:|---------:|--------|
| 1 | **66.7 %** | 4,527,401,216 | 74,036 | 61.2 | `cublas internal::gemvx::kernel<__half,__half,__half,float>` (decode-loop projections) |
| 2 | 9.3 % | 628,577,056 | 19,200 | 32.7 | `cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_tn_align8` (predictor / batch GEMM) |
| 3 | 8.8 % | 595,949,920 | 1,920 | 310.4 | `cublas gemvx __half→float` (LM-head decode path, F16 fallback) |
| 4 | 5.8 % | 391,264,128 | 4 | 97 815 | `ominix_cuda::causal_conv_transpose1d_f32_kernel` (codec upsampler) |
| 5 | 2.8 % | 192,064,672 | 19,200 | 10.0 | `cublas gemvx __half` (smaller projection variant) |
| 6 | 2.3 % | 156,739,168 | 66,399 | 2.4 | `rmsnorm_f16_g32_kernel` (custom) |
| 7 | 1.2 % | 84,588,672 | 16,062 | 5.3 | `attn_decode_gqa_kernel` (custom GQA attention, NOT cuDNN FMHA) |
| 8 | 0.5 % | 31,866,944 | 32,124 | 1.0 | `rope_neox_f16_kernel` (custom) |
| 9 | 0.5 % | 31,732,480 | 32,124 | 1.0 | `add_f16_kernel` (residual) |
| 10 | 0.5 % | 30,784,032 | 12 | 2 565.3 | `dilated_causal_conv1d_im2col_f32_kernel` (codec) |

GPU busy: **6.79 s out of 10.24 s observed** = **33.7 % idle on the GPU**. Inter-kernel gap: median 2.66 us, p90 7.07 us, totalling **1.02 s of small (<100 us) gaps** — pure launch-overhead between micro-kernels.  
CUDA-Runtime API call counts: **264,376 cudaLaunchKernel** + **52,527 cudaMemcpyAsync** (708 ms / 80 % of API time on the latter). Default daemon launch did **NOT** enable Phase 2.5 graphs (`use_cuda_graphs_=false`); a re-run with `TALKER_USE_CUDA_GRAPHS=1` showed **133 `cudaGraphInstantiate` + 133 `cudaGraphLaunch` per request** (one capture per token position) and **no measurable wall-clock improvement** — graphs are rebuilt every step, so launch cost is just shifted to capture time.

## 2. Per-stage breakdown (one full warm request, 7.14 s wall)

| Stage | Wall | GPU busy | Idle % | Notes |
|-------|-----:|---------:|------:|-------|
| Prefill (text embed + first-pos forward) | ~0.36 s | ~0.41 s | low | Single forward over short prompt; wmma dominates briefly |
| **LM autoregressive decode loop** | **~6.17 s (86 %)** | 4.50 s | **27 %** | 35+ gemv/token × N tokens × 2 nets (Talker + Predictor); launch-bound |
| Codec / SpeechTokenizerDecoder (cuDNN+custom conv) | ~0.51 s | 0.51 s | ~0 % | Saturated; large kernels (97 ms causal\_conv\_transpose1d, 2.6 ms dilated\_causal) |

## 3. Memory bandwidth utilisation

GB10 spec: **546.1 GB/s**.  
Median decode gemv = 38 us; at peak BW that window can move 20.7 MB. A typical 1024×1024 f16 weight read = 2 MB → **~10 % effective HBM BW** on the steady-state decode kernel. D2D memcpy total 49.7 ms / 732 MB = **14.7 GB/s** (KV-cache movement, negligible). The pipeline is **launch-bound and SM-occupancy-bound, NOT bandwidth-bound** on GB10. There is large unused HBM headroom that bigger-batch or fused kernels could turn into wall-clock.

## 4. Ranked candidate wins

| Rank | Candidate | Expected payoff | Effort | Risk |
|------|-----------|----------------:|-------|------|
| **1** | **Persistent CUDA Graph re-launch** (capture **once**, launch N times instead of capture-per-pos) | **−25 to −35 % decode wall (≈ −1.5 to −2.0 s)**; closes the 27 % decode-loop idle gap and eliminates ~75 % of 264 k cudaLaunchKernel | Medium (Phase 2.5 already wires capture; needs cache key by `pos<MAX_PREFILL` shape and pointer rebinding; KV-cache pointer becomes a graph node update) | Med — graph-update API on sm_121a, pointer indirection for KV writes |
| 2 | **Fuse Q/K/V projections** + **fuse gate/up SwiGLU** into single cuBLAS-LT batched GEMM per layer | −15 to −20 % decode (drops 3 gemv → 1 per layer × 5 layers) | Medium (cuBLAS-Lt batched / strided-batch with bias epilogue) | Low |
| 3 | **FP8 on Talker FFN gate/up/down + Predictor projections** (extending Phase 2.6 from LM-head only) | −5 to −10 % decode; helps SM occupancy when fused with #2 (FP8 tensor-cores on Blackwell are 2× FP16) | Medium-High (per-layer scale calibration + accuracy gate; existing LM-head cublasLt scaffold reusable, see `talker_cuda_engine.cpp:1093+`) | Med (quality regression) |
| 4 | **cuDNN Frontend SDPA / FMHA replace `attn_decode_gqa_kernel`** | −1 to −2 % (attention only 1.2 % of GPU); skip unless free-with-#1 | Low | Low |
| 5 | **`cudaMallocAsync` + memory pool** (currently 13 cudaMalloc + 6 cudaFree per init, plus 52 k cudaMemcpyAsync — 80 % of API time) | Init-only win; D2D 56 ms is small. Not a steady-state lever | Low | Low |
| 6 | **Predictor speculative / multi-token prefill** (currently 1 token per LM forward) | Potentially 2–3× decode if model-architecture-friendly; needs draft model or self-speculation | Very high | High (correctness) |
| 7 | Codec optimisation (cuDNN frontend for `causal_conv_transpose1d` group) | −2 to −3 % overall (codec is only 7.1 % of wall) | Medium | Low |

## 5. Single first-thing-to-try recommendation

**Convert the Phase 2.5 per-pos `cudaGraphInstantiate`-per-step into a steady-state graph that is captured once (or once-per-shape-class) and re-launched.** This is the closest analogue to the Ascend "abstraction-break" arc: today the engine pays full launch cost on every step (264 k `cudaLaunchKernel`) **and** pays graph-instantiate cost on top when graphs are enabled. The decode loop has 27 % GPU-idle from launch gaps and ~10 % effective BW use — both are launch-overhead-limited, not arithmetic- or BW-limited. A reusable graph eliminates the cudaLaunchKernel per kernel during the loop and turns each token into one `cudaGraphLaunch` (~5 us). Projected: **−1.5 to −2.0 s on a 6.5 s warm run** (RTF ≈ 0.45). It also unblocks #2 and #3, because the savings only show up once launch overhead stops dominating. After that, fused QKV + per-layer FP8 are the natural next-bucket wins; FP8 alone, applied without fixing launch overhead, is unlikely to recover its full theoretical gain because ~one-third of decode wall is sitting on the host.

We are **not yet** at fundamental limits: GB10 is delivering ~10 % of its bandwidth and ~30 % idle. The GB10 arc has ~2× headroom remaining before bandwidth or arithmetic starts to bite. Beyond that, the limit is set by the autoregressive token rate (~150 tokens/s for a 10.24 s clip) and only multi-token-step techniques (#6) can break it.
