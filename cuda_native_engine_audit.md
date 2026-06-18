# CUDA Native QIE-Edit Engine — Readiness Audit

**Auditor:** Claude (Opus 4.7, 1M context)
**Date:** 2026-04-29
**Scope:** Read-only audit of `tools/qwen_image_edit/native/image_diffusion_cuda_engine.{h,cpp}` on `zgx-5b44` to decide if the native CUDA engine is production-ready, partial scaffold, or dead code.
**Sources read on zgx-5b44:**
- `/home/user1/ominix-cuda/tools/qwen_image_edit/native/image_diffusion_cuda_engine.h` (343 LOC)
- `/home/user1/ominix-cuda/tools/qwen_image_edit/native/image_diffusion_cuda_engine.cpp` (1,654 LOC)
- `/home/user1/ominix-cuda/tools/qwen_image_edit/native/cuda_kernels/dit_kernels.{cu,h}` (813 + 188 LOC)
- `/home/user1/ominix-cuda/tools/qwen_image_edit/native/test_image_diffusion_cuda_{init,block,dit,e2e}.cpp`
- `/tmp/qie_3p3c_*.log`, `/tmp/qie_3p3c_*.f32.bin` (prior run artefacts)
- ran `test_image_diffusion_cuda_init` and `test_image_diffusion_cuda_block` at 1024² (PASS, finite, no NaN/Inf)

---

## Verdict at a glance

**Branch B — engine is REAL and END-TO-END FUNCTIONAL but ~2× SLOWER than the CLI baseline at 1024².**

- Every method declared in the header has a real implementation in the `.cpp`.
  `init_from_gguf` → `compute_t_emb_` → `forward_block` (60 blocks of cuBLAS GEMM + custom RMSNorm + multi-axis NEOX RoPE + naive softmax attention + AdaLN modulate + GELU MLP + gated residual) → `forward_dit` → `denoise` (host-orchestrated 20-step Euler-flow loop with patchify/unpatchify) all run through to a finite F32 output latent.
- Smoke evidence is on disk: `/tmp/qie_3p3c_run.log` shows a successful **1024² × 20-step** end-to-end run on 2026-04-26 producing `qie_3p3c_latent.f32.bin` (NaN=0, Inf=0, std=1.034). I re-ran `test_image_diffusion_cuda_init` and `test_image_diffusion_cuda_block` today (2026-04-29): both PASS, finite outputs, init uploads 38.06 GiB of weights cleanly.
- **Performance: 33.8 s/step at 1024²** (per-step log) — **slower** than the ggml-cuda CLI baseline of 15 s/step that the perf project is trying to beat. Native engine is NOT a free win; it is a different point in design space.
- **Quality is unverified.** Final latent magnitudes barely move across 20 steps (max_abs 4.795 → 4.898 ≈ 2 % drift) which suggests near-pass-through behaviour, i.e. the proj_out velocity prediction is close to zero. A VAE decode + visual eye-check has never been done for the 1024² CUDA path. The Phase-3.4d code comment in `denoise()` claims a similar bug was previously fixed for the Ascend twin, but there is no parity log proving the CUDA fix landed.
- **Source files are NOT tracked in git.** `git status` reports `tools/qwen_image_edit/native/` as Untracked. There is no commit on the CUDA repo's `main` for any of the 1,997 LOC of native engine code. The most recent CUDA commit (`9af35f05`) is the in-flight ggml-cuda allocator fix (#187). All "Phase 3.x" commit history in the git log refers to the Ascend twin under `OminiX-Ascend`.

The header comment at `image_diffusion_cuda_engine.h:33` says "Phase 3.1 skeleton + Phase 3.2 will fill + Phase 3.3 will fill". **That comment is stale.** The actual `.cpp` has Phase-3.1 + 3.2 + 3.3a + 3.3b + 3.3c + 3.3b widening (F32 attention path, §5.5.46 Ascend BF16 analog) + 3.4d (Euler velocity-vs-denoised fix) all landed.

---

## Header status

**Dispatch-mapping table (h:14–22):** declared mappings vs implementation in cpp:

| Ascend op | CUDA mapping declared | Actually used in cpp |
|---|---|---|
| `aclnnMm` | `cublasGemmEx` | YES — every projection (mod / qkv / out / mlp_0 / mlp_2 / norm_out / proj_out / img_in / txt_in) is `cublasGemmEx` with F16 weights, F32 accumulate, mostly F32 output |
| `aclnnRmsNorm` / LayerNorm | custom CUDA kernel | YES — `launch_layernorm_noaffine_f32` + `launch_rmsnorm_head_f32_g32` |
| `aclnnApplyRotaryPosEmbV2` | custom CUDA (joint RoPE) | YES — `launch_rope_neox_3axis_f32` + persistent `pe_cos_dev_/pe_sin_dev_` table built at init from temporal=16 + h=56 + w=56 axes |
| `aclnnFusedInferAttentionScoreV2` | cuDNN FMHA / CUTLASS FMHA | **NO** — uses `attn_joint_naive_f32_kernel`, naive O(seq²) softmax. cuDNN handle is created but unused for FMHA |
| `aclnnWeightQuantBatchMatmul` (A8W8) | cuBLAS INT8 / Q8_0 dequant | **NO** — Q8_0/Q4_0 weights are dequantized to F16 at init by `upload_tensor_f16` and stored as plain F16. No on-the-fly INT8/A8W8 GEMM. |
| `aclmdlRI` (ACL Graph) | `cudaGraph` | **NO** — no `cudaGraph_t` capture/replay anywhere. Each step issues 60×~20 launches per block from host. |

**Phase markers in header:**
- h:33 "Phase 3.1 (THIS) lands the engine class skeleton" — stale; everything below is also landed
- h:38 "Phase 3.2 will fill `forward_block`" — DONE in cpp
- h:42 "Phase 3.3 wires the full loop + lm_head/proj_out + VAE decode hand-off" — denoise is DONE; **VAE decode is NOT in the engine** and is delegated to the e2e harness caller
- h:54 "frozen for Phase 3" config struct — present and used

---

## Method-by-method classification

| Method | Header line | Cpp line | Status | Notes |
|---|---|---|---|---|
| `init_from_gguf` | h:114 | cpp:238 | **COMPLETE** | Opens DiT GGUF via `gguf_init_from_file`, validates `general.architecture=='qwen_image'`, uploads all 60 blocks × ~30 tensors + global head/tail (img_in / txt_in / txt_norm / time_lin1 / time_lin2 / norm_out / proj_out) → 38.06 GiB F16 weights on device, builds 3-axis RoPE pe-table, allocs t_emb scratch. `nonfinite_weight_count_=0` confirmed on real run. LLM/vision/VAE paths accepted but ignored (`(void)llm_path;`) |
| `build_pe_table_` | h:269 | cpp:602 | **COMPLETE** | Calls host `build_qwen_rope_pe_host_3axis` (mirrors Ascend `compute_qwen_rope_pe_host`), uploads to F16 device buffers |
| `ensure_scratch_` | h:268 | cpp:617 | **COMPLETE** | Lazy alloc for img/txt residual/norm/q/k/v/attn/mlp/proj scratch in both F16 and F32 (Phase 3.3b widened path) |
| `compute_t_emb_` | h:270 | cpp:692 | **COMPLETE** | Sinusoidal[256] → time_lin1 GEMM → SiLU → time_lin2 GEMM → SiLU, all on device |
| `forward_block` | h:135 | cpp:777 | **COMPLETE** | Real per-block compute: AdaLN-mod, LayerNorm1, QKV F32-out GEMMs, head-wise RMSNorm, multi-axis RoPE, **naive O(seq²) F32 attention**, output proj, gated-residual add, LayerNorm2+AdaLN, MLP_0+GELU+MLP_2, gated-residual add #2. F32 residual chain (Phase 3.3b widened path mirrors Ascend §5.5.46 BF16 fix). H2D/D2H per call. ~1.1 s wall at 1024² seq_tot=4352 |
| `final_proj` | h:144 | cpp:1127 | **STUB / DEAD** | `std::abort()` with "Phase 3.3 stub" message. **Replaced by inline tail logic in `forward_dit`** — never called by the e2e flow |
| `forward_dit` | h:163 | cpp:1147 | **COMPLETE** | 60-block loop with host F32 ping-pong buffers + AdaLN-final tail (norm_out.linear → split shift/scale → LN → modulate → proj_out → patch_out F16 → host F32). Optional `OMINIX_CUDA_DUMP_BLOCKS=1` per-block diagnostic |
| `denoise` | h:200 | cpp:1382 | **COMPLETE** | One-shot txt_norm + txt_in GEMM hoisted before loop; per-step host patchify → device img_in GEMM → forward_dit → host unpatchify → flow-Euler update `x += (x-denoised)/sigma * dt`. Phase 3.4d comment notes the velocity-vs-denoised semantics fix landed. Ref-latent path declared but only implemented for `ref==null` |

---

## Smoke test inventory

**Tests in source:** init / block / dit / e2e — all four buildable, all four built:
```
/home/user1/ominix-cuda/build/bin/test_image_diffusion_cuda_init   (built 2026-04-29)
/home/user1/ominix-cuda/build/bin/test_image_diffusion_cuda_block  (built 2026-04-29)
/home/user1/ominix-cuda/build/bin/test_image_diffusion_cuda_dit    (built 2026-04-29)
/home/user1/ominix-cuda/build/bin/test_image_diffusion_cuda_e2e    (built 2026-04-29)
```

**Tests run today (2026-04-29):**
- `test_image_diffusion_cuda_init <Q8_0 gguf>` → **PASS**. `Phase 3.3a init OK  uploaded=38.06 GiB  nonfinite=0  pe_total_pos=8537`. load_ms=129647 (most of which is GGUF dequant + H2D).
- `test_image_diffusion_cuda_block <Q8_0 gguf>` → **PASS**. Single-block 1024² fwd: `wall_ms=1095.8`, `img_out max_abs=6.158e+07 NaN=0 Inf=0`. (Magnitudes are pre-AdaLN-final; Phase-3.3a smoke does not gate them.)

**Prior run artefacts on box (2026-04-26):**
- `/tmp/qie_3p3c_run.log` — full **1024² × 20-step** e2e run: 676.2 s wall, 33.8 s/step, finite final latent, NaN=0
- `/tmp/qie_3p3c_256_full.log` — 256² × 20-step e2e: 16.0 s wall, 0.78 s/step, finite, NaN=0
- `/tmp/qie_3p3c_latent.f32.bin` — final latent F32 [1, 16, 128, 128], 1 MiB
- **No PNG output anywhere.** VAE decode is delegated to the e2e harness caller and was never wired. Visual correctness unverified.

---

## Performance table (this matters for the verdict)

| Path | 1024²×20 step time | Status |
|---|---|---|
| ggml-cuda CLI (current, what perf project is optimizing) | 15.0 s/step | Baseline |
| torch.compile reference | 5.6 s/step | Target (2.7× gap) |
| **Native CUDA engine (this audit)** | **33.8 s/step** | **2.25× SLOWER than CLI baseline** |

The naive O(seq²×head_dim) attention kernel at seq_tot=4352 is the obvious culprit. cuDNN FMHA + cudaGraph are declared in the header dispatch table as the targets but **never implemented**. The Phase 3.3a probe path (block-only at seq_tot=4352) takes 1.1 s wall × 60 blocks = 66 s of pure compute, which is consistent with the per-step number once you subtract H2D/D2H+patchify overhead.

---

## Why "Branch B, not Branch A"

Native engine is REAL but it is **not yet a perf win**. Compared to the Ascend twin (which has parity bugs but acceptable speed), the CUDA twin has:
- ✅ Numerical health (no NaN, end-to-end run completes)
- ❌ FMHA / FlashAttention (the single biggest CUDA-side perf lever)
- ❌ Quantized GEMM (A8W8 / Q8_0 stays-quantized — current code dequantizes to F16 at load → 38 GiB resident, no compute speedup)
- ❌ cudaGraph capture (60-block per-step launch overhead is paid on every step)
- ❌ VAE decode (engine emits a latent, e2e harness calls a stub)
- ❌ Quality verification (no PNG; pass-through-looking magnitude trace is suspicious)
- ❌ Git tracking (entire engine is `Untracked` — one `rm -rf` and it's gone)

To make this Branch A you would need to:
1. **cuDNN FMHA or CUTLASS FlashAttention.** Expected: replace the 4352² softmax kernel; should drop attention from O(N²) to ~O(N log N) memory traffic. Realistic 5-10× attention speedup; net per-step probably 8-12 s/step. Cost: 3–5 days.
2. **cudaGraph capture of forward_block.** Topology is stable across blocks (60 identical block shapes, just weight pointer swap). Cost: 1–2 days.
3. **Resident-quantized Q8_0/Q4_0 GEMM.** Either dequant-at-runtime per tile (smaller HBM footprint, similar compute) or true INT8 cuBLAS. Cost: 3–5 days.
4. **VAE decode + PNG eye-check.** Borrow VAE from CLI path, validate the 20-step 1024² output is a recognizable cat-edit. Cost: 1 day.
5. **CLI flag `--engine native` + weight-handoff plumbing + git commit the source.** Cost: 1 day.

Total: ~10–15 engineer-days to make native a real Branch-A on CUDA. The 2.7× gap to torch.compile would likely be **closed and possibly beaten** if all four perf items land — torch.compile uses Triton FlashAttention + CUDA graphs + fused norms, all of which the native path could match natively.

---

## Recommendation

**Continue the kernel-fusion / FP8 stack on the CLI path for now, but file a tracking ticket to revive the native engine when CLI hits its ceiling.**

Concrete next step: **run `test_image_diffusion_cuda_e2e` once with the existing fixtures, pipe the latent through the CLI's VAE decode step, and produce a single 1024² PNG.** This is one shell pipeline (`test_image_diffusion_cuda_e2e ... && python decode_vae.py /tmp/qie_3p3c_latent.f32.bin out.png`) and gives a yes/no verdict on whether the engine produces visually correct output. If YES → native becomes a viable Branch-A target once perf items land. If NO → native is parked behind a documented quality bug and CLI perf work is unambiguously the right place to spend time.

After that one decode, also `git add tools/qwen_image_edit/native/` and commit so 1,997 LOC of working code stops being a `git clean -fd` accident waiting to happen.

---

## Files / paths referenced

- `/home/user1/ominix-cuda/tools/qwen_image_edit/native/image_diffusion_cuda_engine.h`
- `/home/user1/ominix-cuda/tools/qwen_image_edit/native/image_diffusion_cuda_engine.cpp`
- `/home/user1/ominix-cuda/tools/qwen_image_edit/native/cuda_kernels/dit_kernels.cu`
- `/home/user1/ominix-cuda/tools/qwen_image_edit/native/cuda_kernels/dit_kernels.h`
- `/home/user1/ominix-cuda/tools/qwen_image_edit/native/test_image_diffusion_cuda_{init,block,dit,e2e}.cpp`
- `/home/user1/ominix-cuda/tools/qwen_image_edit/CMakeLists.txt`
- `/home/user1/ominix-cuda/build/bin/test_image_diffusion_cuda_{init,block,dit,e2e}` (built 2026-04-29)
- `/tmp/qie_3p3c_run.log`, `/tmp/qie_3p3c_256_full.log`, `/tmp/qie_3p3c_256_smoke.log` (prior runs on zgx-5b44)
- `/tmp/qie_3p3c_latent.f32.bin` (most recent 1024² final latent)
