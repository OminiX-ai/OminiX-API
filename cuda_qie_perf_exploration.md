# CUDA QIE-Edit Perf Exploration on GB10 (sm_121, Blackwell)

**Read-only investigation. No code changed.** Profiled the production
ggml-cuda CLI (`~/ominix-cuda/build/bin/ominix-diffusion-cli`, commit
`ad5ef19c`) on host `zgx-5b44`, with the canonical cat→B/W run at
1024² × 5 steps (Q4_0 DiT, Q8_0 Qwen2.5-VL, fp32 VAE, `--diffusion-fa`).

## Run summary

| metric | value | source |
| --- | --- | --- |
| sampling wall time (5 steps × 2 cfg) | **75.58 s** | `/tmp/nsys_qie/run.log:4065` |
| per-step wall (cfg sequential) | **15.12 s** | same |
| denoise GPU window | 80.7 s | sqlite `MAX(end)-MIN(start)` |
| GPU kernel time | **78.24 s (96.9% busy)** | `SUM(end-start)` over `CUPTI_ACTIVITY_KIND_KERNEL` |
| total kernel launches in window | **74,127** (~14,800 / step) | sqlite |
| `cudaLaunchKernel` API time | 57.6 s | nsys api-sum |
| CUDA Graph instantiations | **1** (the Qwen2.5-VL conditioner only — DiT is eager) | api-sum |
| memcpy D2D | 86 GB total / **1.13 s** = ~76 GB/s | mem-sum |
| memcpy H2D | 20.7 GB / 0.39 s | mem-sum |

**For comparison**, the in-tree PyTorch diffusers reference
(`qie_cuda/src/bench_qie_diffusers.py`, `torch.compile max-autotune`,
`bf16`, same input) runs the same workload at **2.80 s / transformer
call** = 5.60 s / step, vs ggml-cuda at **15.12 s/step** — a **2.7×**
gap. PyTorch is the correct ceiling reference; ggml-cuda is leaving a
lot on the table.

## Top 15 kernels by GPU time (5-step run)

| # | kernel (truncated) | GPU s | calls | % of kernel time |
| - | ------------------ | ----- | ----- | ---------------- |
| 1 | `flash_attn_ext_f16<128,128,64,1,...>`                | 15.998 |    600 | **20.45 %** |
| 2 | `mul_mat_q<Q4_0, 128>` (large)                        | 10.823 |  2,900 | **13.83 %** |
| 3 | `k_bin_bcast<op_add, f32>`                            | 10.749 | 16,392 | **13.74 %** |
| 4 | `mul_mat_q<Q4_1, 128>`                                |  6.139 |    580 |  7.85 % |
| 5 | `scale_f32`                                           |  5.230 |  7,860 |  6.68 % |
| 6 | `k_bin_bcast<op_mul, f32>`                            |  4.451 |  7,448 |  5.69 % |
| 7 | `cpy_scalar<f32→f32>` (i.e. `ggml_cont`)              |  4.053 |  4,923 |  5.18 % |
| 8 | `quantize_mmq_q8_1` (activation quant pre-mmq)        |  3.175 |  7,170 |  4.06 % |
| 9 | `unary_op_kernel<op_gelu, f32>`                       |  2.098 |  1,191 |  2.68 % |
|10 | `cpy_scalar_contiguous<f32 → __half>` (FA K/V cast)   |  1.923 |  1,800 |  2.46 % |
|11 | `im2col_3d_kernel<__half>` (VAE encode/decode)        |  1.793 |     57 |  2.29 % |
|12 | `k_bin_bcast<op_repeat>`                              |  1.599 |  2,400 |  2.04 % |
|13 | `concat_f32_dim2`                                     |  1.584 |  1,800 |  2.02 % |
|14 | `concat_f32_dim0`                                     |  1.520 |    656 |  1.94 % |
|15 | `rms_norm_f32<256, true, false>`                      |  1.235 |  2,400 |  1.58 % |
|—  | (`norm_f32` LayerNorm, `pad_f32`, etc. follow)        |        |        | rest 7 % |

(Source: `nsys profile` artefact `/tmp/nsys_qie/qie_5step.{nsys-rep,sqlite}` on `zgx-5b44`.)

### What this says

1. **Attention is f16 with f32-accum (FA enabled)** — `flash_attn_ext_f16<128,128,64>` at 20%. 600 calls = 60 layers × 10 transformer calls. This is the *Ampere-style* (mma.sync 16x8x16 fp16) FA kernel. **There is no Blackwell-tuned FA-3 path**; ggml-cuda has no awareness of sm_100/sm_120 tensor-memory or wgmma. Each FA call averages 26.6 ms, working on ~9k tokens × 24 heads × 128 d_head.
2. **Q4_0 mmq dominates matmul** — 2,900 launches × 3.7 ms = 10.8s on `mul_mat_q<Q4_0,128>`. Plus `mul_mat_q<Q4_1,128>` (6.1s, 580 calls — that's 1 per layer per cfg = MLP `to_out`) and Q8_1 activation prequant (3.2s). **The whole diffusion model is Q4 — no FP8 path exists**, even though sm_121 has native E4M3 support.
3. **Element-wise kernels dwarf matmul** — `add` (10.7s) + `mul` (4.5s) + `scale` (5.2s) + `cpy` (4.0s) + `quant` (3.2s) + `cont→half` (1.9s) + `repeat` (1.6s) + `concat` (3.1s) + `gelu` (2.1s) = **36.3 s = 46% of GPU time** spent on bandwidth-bound glue between matmuls. Each is a separate kernel launch.
4. **Memory bandwidth is *not* the binary bottleneck**: only 1.13s of pure-memcpy time. But the elementwise kernels above ARE bandwidth-bound — they read/write the activation tensor (~64 MB at hidden 3072 × 9k tokens fp32) repeatedly. Those 16,392 `add` launches *are* the memory traffic.
5. **CUDA Graphs not used for DiT** — only 1 graph instantiation in the entire 75s window (it's the Qwen2.5-VL conditioner; DiT is rebuilt eagerly each transformer call). 14,800 launches/step × ~3.5 µs API overhead each ≈ **52 ms/step pure launch overhead** baked in.
6. **No fused RMSNorm-modulate-gate kernel exists.** Per DiT block (`qwen_image.hpp:255-345`) the modulation+norm+gate sequence is 13 separate ggml ops. Multiplied by 60 blocks × 10 calls = 7,800 unfused launches — most of items 3, 5, 6 above.

GB10 unified memory peak BW is ~120 GB/s (LPDDR5x). Achieved on the elementwise sweep: with 36 s on ~Y GB of activation traffic per layer (~120 MB/elementwise × 16k launches ≈ 1.9 TB total), achieved ≈ **53 GB/s on the elementwise tail = ~44% of peak**. For Q4_0 mmq, 11 GB of weights × 600 reads ≈ 6.6 TB at 10.8s = **610 GB/s effective** (well above the unified-memory ceiling — kernel is running mostly out of L2 + register-resident dequant; this is compute-bound on Q4-decode + tensor-core throughput for a quantized GEMM that doesn't use Blackwell's wgmma).

## Ranked candidate wins

| Rank | Candidate | Expected win | Risk | Effort | Where |
| ---- | --------- | ------------ | ---- | ------ | ----- |
| 1 | **Capture DiT into a CUDA Graph** (cudaStreamBeginCapture around 60-block forward, replay each step / each cfg branch) | **Eliminate ~50 ms/step launch overhead = ~5–8% step**; bigger if the elementwise tail is launch-bound, which the 14.8k calls/step and 70% API time on `cudaLaunchKernel` strongly suggest. Could reach **15–25%**. | Low. ggml already has graph infra (`ggml-cuda.cu:2927`); just needs DiT op coverage. PE-cache and modulate-index already shape-stable. | ~200 LOC + debug, 1–2 days | `ggml/src/ggml-cuda/ggml-cuda.cu` graph compatibility checks; `qwen_image.hpp` `build_graph` |
| 2 | **Hand-fused RMSNorm + modulate (scale/shift) + gated-residual kernel** to replace items 3/5/6/7 of the kernel table inside each DiT block | RMSNorm/LN + 2 muls + 1 scale + 1 add per modulated branch × 4 branches × 60 layers × 10 calls ≈ **8–12 s** of elementwise + cont could fold into 60×10×4 = 2,400 launches. Estimate **15–20% step time**. | Medium. Need to audit shape-vs-broadcast for img/txt and Q4 CFG-batched code path. Numerical regression possible in modulate broadcast. | ~600 LOC + tests, 3–5 days | new `.cu` next to `norm.cu`; wire via `ggml_qwen_image_block_modulate` custom op |
| 3 | **FP8 (E4M3) DiT weights via cuBLASLt FP8 GEMM** — replace `mul_mat_q<Q4_0,128>` for Linear layers (Q,K,V,to_out, MLP up/down) with cuBLASLt FP8 GEMM; activations bf16 | Matmul drops from ~17 s (Q4_0+Q4_1+Q8_1 quant) to ~6–8 s on Blackwell tensor cores at 2× density vs bf16. Expect **10–15% step**. **TTS got ~5% from FP8 on lm-head alone; DiT has 60× more FP8-eligible matmul.** Better quality vs Q4 too. | Medium-high. Need calibration (per-tensor scaling) and a CUDA path that hands raw FP8 tensors to cuBLASLt. ggml has no FP8 type today. | ~1,500 LOC (new ggml type, packing, cuBLASLt wrapper), 1–2 weeks | new files `ggml/src/ggml-cuda/fp8/`; add `GGML_TYPE_FP8_E4M3` |
| 4 | **Replace `ggml_ext_attention_ext` cast/permute prologue with one fused QKV-permute-cast-pad kernel; or hand-call cuDNN FMHA / FlashAttention-3** | The FA kernel itself is fine (20%), but the *3 ms of prep* per call (`cpy_scalar f32→f32` + `cpy_scalar_contiguous f32→half` + `pad`) costs ~3 s/run = **3–4%**. Switching to FA-3 (Hopper/Blackwell wgmma) could cut FA itself by ~30% = **~6%**. | Medium. FA-3 needs head_dim 128 path, which *does* exist in upstream Tri Dao branch and in cuDNN 9.x. Need to handle Q4 CFG-batched mask too. | ~400 LOC + cuDNN integration, 3–4 days | `ggml_extend.hpp:1265-1380`; new dispatch in `fattn-mma-f16.cuh` for sm_121 |
| 5 | **Drop `ggml_cont` after attention output split (line 184–185 of qwen_image.hpp)** — the txt/img view-then-cont is inserted because downstream `to_out_0->forward` expects contiguous; with cuBLASLt epilogues we can ingest strided directly | Two `cpy_scalar` per layer per call × 60 × 10 = 1,200 calls, ~2.5 s = **3% step** | Low. Just verify the linear-forward strided-input path in ggml-cuda. | ~50 LOC, half a day | `qwen_image.hpp:184-186`; `mul_mat` strided support |
| 6 | **CFG batching (run cond+uncond as ne[3]=2)** — already partly implemented (`OMINIX_CFG_BATCHED=1`), currently disabled for 1024² because mask footprint > 256 MiB budget | If batch utilisation lifts each kernel by ~1.4× (not 2× because some kernels are already underfull), expect **20–35%** end-to-end. | Low-medium. Mask footprint problem is the actual blocker — need a sparse/blocked mask representation, not dense [N,L_q,L_k] f16. | ~300 LOC + memory plumbing, 2–3 days | `stable-diffusion.cpp:2364-2399`, `OMINIX_CFG_BATCHED` path |
| 7 | **Disable Q8_1 activation pre-quantization for low-token matmuls** — `quantize_mmq_q8_1` runs 7,170× = 3.2 s. Many small matmuls (modulation projection, LayerNorm-to-modparam) re-quantize activations that are tiny. A direct bf16×Q4 dot path would save the round-trip. | Item alone = **3–4%**. | Low. Already a tunable in `mmq.cuh`. | small | `ggml-cuda/mmq.cu` MMQ_MIN_BATCH heuristic |

(Items 1, 2, 6 stack mostly orthogonally; 3 partially overlaps 2.)

## Recommendation: try CUDA Graphs on the DiT first

Reasons:
1. **It is the biggest single concrete win available with the lowest
   risk and effort** (1–2 days vs 1–2 weeks for FP8). The harness is
   already there in `ggml-cuda.cu`; we only need to widen graph
   compatibility for the DiT op set and confirm shape stability across
   denoise steps (which the existing PE-cache work indicates is
   already true).
2. **The profile is unambiguous**: 14,800 launches/step, 96.9% GPU
   busy, only 1 graph instantiation observed → graphs are *not*
   currently active for DiT despite the project's belief. Confirming
   this and turning them on is the highest-value first move.
3. **It de-risks every subsequent optimisation.** Once the DiT runs as
   a captured graph, fused-kernel work (item 2) and FP8 swap-in
   (item 3) become drop-in node replacements rather than
   eager-launch refactors.
4. **It gives the right diagnostic signal** for the bigger question.
   If turning on graphs only buys 8% (just the launch overhead), then
   the bottleneck is genuinely the kernels, and FP8/fusion is
   warranted. If it buys 25%+, the bottleneck was the launch cadence
   itself, which would change the next priority.

The PyTorch `torch.compile` reference at 5.6 s/step shows the workload
admits ~2.7× headroom on this same GPU — strong evidence that the win
is real and not a hardware ceiling.

## Artefacts left on `zgx-5b44`

- `/tmp/nsys_qie/qie_5step.nsys-rep` (6.5 MB) — open in Nsight Systems
- `/tmp/nsys_qie/qie_5step.sqlite` (15 MB) — for further sql queries
- `/tmp/nsys_qie/run.log` — full ggml-cuda log of the profiled run
- `/tmp/nsys_qie/run.sh` — exact command line
- `/tmp/nsys_qie/out.png` — sanity-check output image (5-step degraded but recognisable)
