# QIE-Edit on Ascend 910B — Handoff for Review

**Status (2026-04-29 evening, post-§5.5.13c discovery): contract gate WIDE OPEN.**
Both Ascend QIE paths produce NaN at 256² and 1024² in production cadence. The "GREEN" findings from §5.5.59/65/66/67 + audit #188 + §5.5.13 partial close were all under `SD_FIRST_NAN_TRACE=1`, which serializes per-node compute and adds CPU work between dispatches — that timing change masks the production NaN bug. Real production cadence (no tracer) is fully NaN.

**Maintainer:** Yue Chen (yue.chen@futurewei.com)
**Date:** 2026-04-29 (revised after the §5.5.13c session)

> **READ THIS FIRST**: the older sections of this handoff describe a saga state that has since been invalidated by direct production-cadence verification at HEAD. Specifically, the "first-NaN at node 4654 RMSNorm" finding was a tracer artifact, and the "switch from CLI to native" audit verdict was based on tracer-mode evidence. The corrected reckoning is in the **"Saga reckoning, 2026-04-29 evening"** section near the bottom. Use that section as the canonical state. The earlier sections are kept for archaeological context only.

---

## TL;DR

Qwen-Image-Edit-2509 (DiT) runs end-to-end correctly at 256² on Ascend 910B via ggml-cann. At 1024² it produces all-NaN output. After 60+ rounds of bisect (§5.5.0–§5.5.65), the actual first-NaN node has now been pinpointed:

- **Step 0 (first DiT forward pass): fully clean across all 10,213 nodes.**
- **Step 1 (second DiT forward pass): first-NaN at node 4654, op=`RMS_NORM`, shape `[128, 24, 4096, 1]` f32, with clean input.**

This is `norm_q` or `norm_k` (per-head Q/K RMSNorm) inside `Attention::forward` at roughly DiT block 27. The kernel's own input is clean (0 NaN at the moment of post-compute callback) but its output contains 4096 NaN values — exactly one per token-row. The pattern strongly suggests either an Ascend `aclnnRmsNorm` kernel quirk on the second graph invocation at this exact shape, or a buffer-aliasing issue that the existing `SD_FIRST_NAN_TRACE` callback timing happens to hide.

The next step (§5.5.66, not started) is to dump the actual input buffer at that node and compare against a CPU reference RMSNorm to decide which.

---

## Repository locations

| Where | Path | Branch | Notes |
|---|---|---|---|
| **Production (live)** | `ac03:/home/ma-user/work/OminiX-Ascend` | `main` | **54 commits ahead of `origin/main`, NOT pushed**. This is where §5.5.x work lives and the binary that produced the §5.5.65c trace was built here. |
| Mac local mirror | `/Users/yuechen/home/OminiX-Ascend` | — | Separate fork-style branch with the same §5.5.65 work landed independently (commit hashes differ: `b8d36560`, `5e9bd3e1`, `b2dd5560`). Useful for code reading on Mac; **the trace evidence is on ac03**. |
| Upstream remote | `https://github.com/ymote/OminiX-Ascend.git` | `main` | Last upstream commit is `7306b7e5 Fix CANN9 Qwen3.6 frontend shutdown`. The §5.5.x saga work has not been pushed. |

**ac03 access:** SSH config alias `ac03` (Huawei ModelArts notebook, user `ma-user`). NPU driver: `npu-smi 23.0.6`. Hardware: 910B4 (32 GB HBM).

---

## What's in production code on ac03 (saga commits, newest first)

```
a078106  qie(§5.5.65): restore §5.5.53b INPUT flag annotations clobbered by scp
1e2de75  qie(§5.5.65): count true NaN only, not Inf
efd5384  qie(§5.5.65): extend FIRST-NAN trace to F16/BF16 + skip non-scannable
a1ca71d  qie(§5.5.65): add SD_FIRST_NAN_TRACE per-node post-compute NaN scan
737e7c3  qie(Q2.4.5.5.60): § doc — d82ba89 verification PARTIAL, 256² still RED
d82ba89  fix(qie-edit): §5.5.59 — protect INPUT leaves from gallocr free + reuse
82e6e1a  fix(qie-edit): mark model_out as OUTPUT — partial close on 1024² multi-step NaN
c88ca9b  fix(qie-edit): partial graph-allocator alias fix — closes 256² NaN, 1024² still RED
```

The `§5.5.x` numbering is internal to `docs/qie_q2_phase4_smoke.md` (the saga journal — read it for full chain of evidence).

### Files modified across the saga (cumulative)

| File | What changed | Why |
|---|---|---|
| `ggml/src/ggml-alloc.c` | `+24 / -0` lines. Added INPUT-flag protection so leaf input tensors aren't freed/reused mid-graph by gallocr. Recursive view-chain traversal so views of INPUT tensors are also protected. | §5.5.59 closed 256² NaN; §5.5.61–62 extended coverage. |
| `tools/ominix_diffusion/src/ggml_extend.hpp` | `+294 / -134` lines. Added `SD_FIRST_NAN_TRACE` env-gated per-node post-compute eval_callback that scans every tensor immediately after its op completes. F16/BF16 aware, counts true NaN only (skips ±Inf). | §5.5.65 — needed because `SD_NAN_CHECK` reads at end of graph were corrupted by gallocr slot reuse. |
| `tools/ominix_diffusion/src/qwen_image.hpp` | `+7 / -0` lines. Marks `model_out` and other residual-stream tensors as `OUTPUT` so gallocr can't recycle their slots. | §5.5.56–58 partial close; surgical OUTPUT marking. |
| `tools/ominix_diffusion/src/denoiser.hpp` | `+11 / -0` lines. Per-step latent precision instrumentation. | §5.5.50–51 step-count sweep + sampler trace. |
| `docs/qie_q2_phase4_smoke.md` | `+376 / -0` lines. Saga journal — chain of evidence for every §5.5.x. | Documentation only. |

### Working tree on ac03 (uncommitted)
```
M  ggml/src/ggml-alloc.c                                    # in-progress §5.5.59 tweak
?? ggml/src/ggml-alloc.c.bak_5561                           # backup of pre-§5.5.61
?? tools/probes/qie_block0_cpu_reference/...                # probe artifacts (gitignored OK)
?? tools/probes/qie_q45_real_denoise_smoke/...              # probe artifacts
?? tools/probes/qie_q45_step4_full_denoise/...              # probe artifacts
```

---

## Reproducing the §5.5.65c trace (the authoritative evidence)

Binary: `/home/ma-user/work/OminiX-Ascend/build-w1/bin/ominix-diffusion-cli` (built 2026-04-29 07:06).

Models on ac03:
```
/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf
/home/ma-user/work/qie_weights/Qwen2.5-VL-7B-Instruct-Q4_0.gguf
/home/ma-user/work/qie_weights/mmproj-BF16.gguf
/home/ma-user/work/qie_weights/split_files/vae/qwen_image_vae.safetensors
```

Run:
```bash
SD_FIRST_NAN_TRACE=1 /home/ma-user/work/OminiX-Ascend/build-w1/bin/ominix-diffusion-cli \
  -m  /home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf \
  --qwen2vl /home/ma-user/work/qie_weights/Qwen2.5-VL-7B-Instruct-Q4_0.gguf \
  --qwen2vl-vision /home/ma-user/work/qie_weights/mmproj-BF16.gguf \
  --vae /home/ma-user/work/qie_weights/split_files/vae/qwen_image_vae.safetensors \
  -p "a cat" -W 1024 -H 1024 --steps 3 --seed 42 \
  -o /tmp/out.png 2>&1 | tee run.log
```

Existing log of this run: `ac03:/home/ma-user/work/qie_5565c/run.log`.

### Expected output (what was actually observed)

```
[INFO ] qwen2.5vl: tracing 1154 nodes one-at-a-time
[INFO ] qwen2.5vl: no NaN found across 1154 nodes (SD_FIRST_NAN_TRACE)
[INFO ] [NaN CHECK] encoder/cond.c_crossattn: OK (32256 elements, range=[-142.4, 97.6])
[INFO ] qwen_image: tracing 10213 nodes one-at-a-time
[INFO ] qwen_image: no NaN found across 10213 nodes (SD_FIRST_NAN_TRACE)        ← step 0 clean
  |================>                                 | 1/3 - 737.86s/it
[INFO ] qwen_image: tracing 10213 nodes one-at-a-time
[ERROR] qwen_image: node 4654/10213 op=RMS_NORM name='node_4654' shape=[128,24,4096,1] type=f32 nans=4096/12582912
[ERROR]   src[0]: op=RESHAPE name=' (view) (reshaped)' shape=[128,24,4096,1] type=f32 nans=0/12582912
[ERROR] qwen_image: aborting at first-NaN node 4654/10213 (SD_FIRST_NAN_TRACE)
[ERROR] diffusion model compute failed
```

**Decoded:**
- 128 = head_dim
- 24 = num_heads
- 4096 = num image tokens at 1024² (1024/16 = 64; 64² = 4096)
- 1 = batch
- The 4096 NaN count = exactly num_tokens, i.e. one NaN per token-row across all heads/positions
- This shape matches the per-head Q-norm or K-norm in attention (`qwen_image.hpp:91-92` declares `RMSNorm(dim_head=128, eps)` for `norm_q`/`norm_k`). With ~170 nodes/block × 60 blocks, node 4654 falls at roughly DiT block 27.

### Why this trace is trustable (and earlier ones weren't)

`SD_FIRST_NAN_TRACE` (§5.5.65) registers a per-node eval_callback that fires immediately *after* each op finishes computing. It reads the output AND each `src[i]` at that exact moment, before gallocr has had a chance to reuse any slot for downstream ops.

Previously, `SD_NAN_CHECK` (the older mechanism) read tensors at *end of graph* — by which time gallocr has reused freed slots many times. That made every "NaN at end-of-graph" reading unreliable as a producer-localization signal. §5.5.63's "node 14 CONCAT" finding came from this contaminated source and was refuted by §5.5.64a's no-ref control test (the reported producer site never even fired in the no-ref run, yet the NaN signature was identical).

There is one earlier `qie_5565b/run.log` that reports a different first-NaN inside `qwen2.5vl` at node 28 ADD, with `leaf_121` showing 903/1849 NaN. **Ignore it.** That run was made against a binary built before commit `efd5384`/`1e2de75` (which fixed the trace's true-NaN-vs-Inf counting); the leaf-121 reading is a false positive from an older buggy trace path.

---

## What 60 rounds of bisect have actually established

The saga isn't a series of failed fixes. Each `§5.5.x` either landed a real fix or ruled out a hypothesis. Net result of the legitimate fixes:

- **Q4_0 dequant correctness (§5.5.42).** Mirrored ggml-cann's `mul_mat_quant` V2 dispatch byte-for-byte. Before this, mid-block residuals were drifting numerically.
- **F16-saturation in residual stream (§5.5.30, §5.5.45–46).** Widened gated-residual + Q/K/V projection outputs to BF16. Closed step-1 NaN at block 27 in single-step decode and most multi-step cases.
- **AdaLN modulation chunk order (§5.5.21, `ce34b9f`).** CPU reference was swapping scale/shift; engine was correct. Fixed the reference.
- **Graph-allocator buffer alias on second compute (§5.5.59 `d82ba89`).** Added INPUT-flag protection in `ggml-alloc.c` so leaf input tensors (latent, RoPE positions) aren't freed/reused mid-graph. **This closed 256² entirely.** Before it, both 256² and 1024² were RED on multi-step.
- **Recursive view-chain INPUT-flag traversal (§5.5.62).** Same mechanism extended to in-graph CONT/RESHAPE views of INPUT tensors.

**State after all that:** 256² works at any step count. 1024² works for step 0 but fails at step 1. The remaining failure is the single RMS_NORM node identified above.

### Hypotheses ruled out (don't reopen)

- F32 attention accum / softmax precision (§5.5.55)
- F16 residual saturation as the *only* path (§5.5.30 fixed it; new failure mode is structural)
- Sampler-bug class (CacheDiT, layer-bisect — §5.5.51)
- ggml-cann concat shape bug at `[64, 8192]` (§5.5.63 retracted by §5.5.64a)
- Quantization correctness — Q4_0/Q4_1/Q5_K dispatch is bit-accurate (§5.5.31, §5.5.42)
- Weight load — `img_mod_1.weight` etc. are bit-identical to CLI baseline (§5.5.31)
- RoPE precision and shape (§5.5.18 oracle GREEN)
- t_emb chain (§5.5.20 oracle GREEN)
- Audio: not relevant (this is QIE-Edit / image stack)

---

## Open work (§5.5.66 — fix track)

Tracked as task #179. **Not started.**

### Important corrections to the §5.5.65c reading (added 2026-04-29 after a 2nd-opinion review)

Three things in the original framing of this handoff turned out to be wrong or imprecise:

1. **`src[0] nans=0` ≠ "clean input."** The `SD_FIRST_NAN_TRACE` callback **skips Inf** when counting (`tools/ominix_diffusion/src/ggml_extend.hpp:2152, 2166` — counts true-NaN only). If src[0] contains Inf values, the count still reads 0. RMSNorm of a row with Inf produces NaN deterministically:
   ```
   mean(x²) = Inf  →  1/sqrt(Inf) = 0  →  Inf × 0 = NaN
   ```
   This mechanism explains the 4096 NaN count *much better* than "per-row variance went to zero," and is consistent with the saga's prior F16-saturation work (§5.5.30, §5.5.45–46) being incomplete for this exact codepath.

2. **The shape arithmetic in this doc was sloppy.** With `[128, 24, 4096, 1]` and RMSNorm along `dim_head=128`, there are `24 × 4096 = 98,304` rows. So "one NaN per row" is not the literal description. 4096 NaN likely means *one head's worth* of NaN values — pointing at one specific head being the upstream Inf-producer. Re-do the per-head magnitude trace (cf. §5.5.34) at step 1 to find which head saturates.

3. **It's not `aclnnRmsNorm`.** For this tensor size, ggml-cann uses a **manual decomposition** (`ggml/src/ggml-cann/aclnn_ops.cpp:1180-1212`): `Mul → Mean → add eps → Rsqrt → Mul`. So "Ascend kernel bug" is the wrong frame — the bug, if in this chain, could be in any of those five ops, with broadcast `Mean` and pool-workspace lifetime as the higher-priority suspects.

### Additional missed coverage class

`ggml/src/ggml-alloc.c:1007` — the gallocr realloc check looks at node/leaf counts and sizes, not flag semantics. OUTPUT/INPUT flag changes (the §5.5.59 fix mechanism) may not trigger a re-plan, meaning **step 1 reuses step 0's allocation plan even after flag annotations are updated.** This is a concrete coverage gap that could explain why §5.5.59 closed 256² but step-1 1024² still aliases — at 256² the smaller offsets happen to coincide harmlessly, at 1024² they don't.

### Revised plan for §5.5.66

1. **Patch `SD_FIRST_NAN_TRACE` to additionally count Inf and max-finite per tensor**, then re-run the §5.5.65c trace. New decision branches:
   - If `src[0]` has nonzero Inf → bug class is *upstream F16-saturation*. Walk back through the producer chain (likely a matmul output or residual still in F16 on this codepath) and find the missed widening. Don't dump RMSNorm input at all; dump the producer.
   - If `src[0]` has 0 Inf and 0 NaN with sane max-finite → bug class is the manual RMSNorm decomposition or an allocator-replay issue. Then proceed to step 2.

2. **Dump correctly.** Don't `ggml_set_output()` a RESHAPE view — the saga already learned views don't pin backing storage (`docs/qie_q2_phase4_smoke.md:3016, 3100`). Use `ggml_cont` to materialize, or pin the producer tensor. Capture finite-class stats *first*, full buffer second.

3. **CPU reference comparison only after Inf is ruled out.** And if both CPU and NPU produce NaN, that does not prove "input is the real culprit" — it proves the dumped input contained Inf or was captured post-aliasing. Map output-NaN indices back to input-Inf/huge indices to localize.

4. **Test the realloc-coverage hypothesis.** Force gallocr to fully re-plan between step 0 and step 1 (e.g. clear cached plans, or alter graph structure trivially between steps to invalidate the size+count check). If 1024² then runs clean to step 1, the bug is a reuse-of-stale-plan issue and the §5.5.66 fix is in `ggml-alloc.c:1007` — not in any kernel.

5. **Cross-check at 768²** (still useful): the same trace should fire at the analogous node, since 768² has 2304 image tokens. If 768² works, the failure is 1024²-specific allocator pressure.

### What NOT to do
- Do not assume the §5.5.65c reading is "authoritative" without first re-running with Inf counting enabled.
- Do not dump a RESHAPE view directly — use `ggml_cont` or pin the producer.
- Do not assume `aclnnRmsNorm` is in play — ggml-cann uses a manual decomposition for this shape.
- Do not assume "step 0 clean" implies kernel correctness — it implies "no NaN," not "no Inf, not pathological."
- Do not reopen broad gallocr instrumentation as a *first* move; first patch the trace, then test the §5.5.66 plan above.
- Do not assume node 14 CONCAT is the producer. §5.5.63 was wrong.

### 2nd-opinion review

Full critique: `/tmp/codex_qie_review.md` (independent reviewer, codex 0.125.0, ran 2026-04-29 11:13–11:17, read-only sandbox). It cites specific lines in `ggml_extend.hpp`, `aclnn_ops.cpp`, and `ggml-alloc.c` for each claim above.

---

## Reference: also-relevant files

- **Saga journal:** `ac03:/home/ma-user/work/OminiX-Ascend/docs/qie_q2_phase4_smoke.md` — full §5.5.x chain.
- **Native engine path** (separate from CLI path used in §5.5.65c): `tools/qwen_image_edit/native/image_diffusion_engine.{cpp,h}` — last cat-PNG output is GREEN at 256², used for harvesting per-block dumps. Note that the §5.5.65c trace ran the **CLI** path, which now matches engine path closely.
- **CLI binary source:** `tools/ominix_diffusion/` (this is what `build-w1/bin/ominix-diffusion-cli` is built from).
- **§5.5.65 instrumentation:** look in `tools/ominix_diffusion/src/ggml_extend.hpp` around lines 2080–2260 for `SD_FIRST_NAN_TRACE` callback + the `ggml_first_nan_check_callback` body. Search for `[FIRST-NAN]`.
- **Working trace logs on ac03:**
  - `/home/ma-user/work/qie_5565c/run.log` ← authoritative §5.5.65c result
  - `/home/ma-user/work/qie_5565b/run.log` ← ignore (pre-fix binary)
  - `/home/ma-user/work/qie_5565/run.log` ← earlier attempt

---

## Honest assessment

The 60-round count looks bad in isolation. The reason it took this long is that the diagnostic infrastructure (`SD_NAN_CHECK` end-of-graph reads) was lying — every "found the bug" claim was based on contaminated evidence. §5.5.65's per-node post-compute callback is the first reading from this saga that can actually be trusted.

With the trustable trace in hand, the picture is much narrower than the saga's length suggests:
- Step 0 across 10,213 ops is fully clean → kernels, weights, RoPE, attention, VAE, graph build, quantization are all correct.
- Step 1 fails at exactly one op → the remaining bug is either one specific kernel quirk on second invocation, or one specific aliasing case the gallocr coverage missed.

Either is fixable in a focused session. The hard part — knowing where to look — is now solved.

— Yue Chen, 2026-04-29

---

## Saga reckoning, 2026-04-29 evening (canonical state)

After dispatching §5.5.13c (#203) to widen `img_mod.1` / `txt_mod.1` matmul outputs to BF16, I discovered the saga is in a much worse state than previous reports indicated. The issues:

### What was actually verified at HEAD `a91bcfb`

- **256² CLI in production cadence (no tracer): 16384/16384 NaN latent → all-black PNG (2301 bytes).**
- 1024² CLI: also NaN (§5.5.67 verification trace showed 4096 Inf in `src[0]` before RMSNorm at node 4654).
- Native engine in production cadence: not directly tested in this session. Inferred broken because:
  - §5.5.13 (#190) said native produces all-NaN without the c_skip+c_out patch.
  - That patch was applied and **then reverted** in #190.
  - HEAD (a91bcfb) does not have it.
  - The "polka-dot at 256²×20" eye-check from #190 was VAE-decode-only of a saved latent from the patched run, not a fresh native forward.

### Why the previous "GREEN" findings were wrong

`SD_FIRST_NAN_TRACE=1` serializes per-node compute (one-node subgraphs with explicit sync between each). This timing change masks the production NaN bug.

- §5.5.59 fix verification at 256² × n=3 → tracer-mode "GREEN".
- §5.5.65c trace at 1024² × 3 → identified "first-NaN at node 4654 RMSNorm" as a tracer-mode artifact.
- §5.5.66 "smoking gun" (Inf-aware tracer flips step-1 RED→GREEN with no compute change) was actually evidence that **the bug is timing-dependent between dispatches**, NOT evidence of an allocator-replay class bug. The codex critique flagged "Inf × 0 = NaN through RMSNorm" as a *more plausible* mechanism in writing before this finding — that critique was itself missed.
- §5.5.67 gallocr re-plan fix (`f695ab4`) was based on the wrong diagnosis. Verification ran in tracer mode and looked GREEN, but production cadence is still NaN.
- Audit #188 ("Branch B switch native to production, 1.17 s/step NaN=0 finite latent") almost certainly also relied on tracer-mode evidence.

### What §5.5.13c (#203) showed

- Agent edited `tools/qwen_image_edit/native/image_diffusion_engine.cpp:3681-3734` to widen `img_mod.1` / `txt_mod.1` matmul output to BF16, plus an in-place BF16→F16 cast back before downstream consumers (mirroring §5.5.45/46 QKV pattern).
- BUT ran `build-w1/bin/ominix-diffusion-cli` for verification. That binary is built from `tools/ominix_diffusion/` (CLI path) — **it does not link the native engine code**.
- So the patch was dead code in the test.
- CLI run produced all-NaN regardless of `QIE_MOD_BF16=1` or unset (confirmed by env-unset re-run).
- The patch is still present on disk (uncommitted) but unverified.

### What's still valid

- **CUDA work (different platform):**
  - #187 CUDA QIE norm-modulate fusion: **+5.85% wallclock at 1024²×20**, bit-identical PNG.
  - #182 CUDA TTS env-flip: **+7% warm 2nd+ requests** (zero code change).
- **Silent race-condition fixes caught by perf agents:**
  - #186: `cudaMemcpyAsync(pos_dev, pos_host_pin)` race in P1's pattern (would have failed silently in production multi-step decode).
  - #193: `rep_penalty_kernel` race when recent-window has duplicate tokens (affected both predictor and existing P2 talker path).
- **Bug isolation (#201, #202):**
  - Per-op F32 references show all weight-free ops match within F16 precision.
  - Q4_0 dequant + matmul oracle confirms `img_mod.1` block 0 (Q5_K) and block 1 (Q4_0) and `ff.net.2` block 1 (Q4_1) match F32 oracle at cos=1.000000. The 491 / 6e5 magnitudes are intrinsic to the trained quantized weights.
  - The fix-shape inference (BF16 widening) was plausible. The mechanism in #203 was wrong (codebase mismatch).

### Action items for the NEXT session — do not dispatch without fresh codex review

1. **Identify which binary actually exercises the native engine** in production. The CLI doesn't. Likely candidates: `test_image_diffusion_cuda_e2e`, `test_image_diffusion_cuda_init`, etc. Map binary → source. This is prerequisite to ANY native engine fix verification.
2. **Build a standalone production-cadence reproducer** for the native engine: invoke `denoise_full()` end-to-end at 256²×20, dump the latent, decode via VAE, check NaN count + visual.
3. **Re-verify EVERYTHING in production cadence** (no `SD_FIRST_NAN_TRACE`). The §5.5.59 / §5.5.66 / §5.5.67 GREEN claims are not trustworthy until re-verified at production cadence.
4. **Re-derive audit #188 verdict** with production-cadence evidence.
5. **Strategic question to escalate**: is shipping at 1024² achievable in this saga's timeline, or is descoping (256² only, Diffusers Python wrapper, etc.) the right move?

### Validation gates raised

Future audit-style strategic decisions require:
- 1024² PNG eye-check **in production cadence (no tracer)**
- VAE decode of a freshly-generated latent (not a saved one)
- Per-step latent stats reported (mean, std, range, NaN/Inf count)
- Artifact paths cited
- One independent codex reviewer pass
- Explicit "production cadence verified" claim before strategic claims

### Working directories on ac03

- Repo: `/home/ma-user/work/OminiX-Ascend` (user `ma-user`). HEAD `a91bcfb`. 56 commits ahead of unpushed `origin/main`.
- §5.5.13c uncommitted patch: `M tools/qwen_image_edit/native/image_diffusion_engine.cpp` (works in spirit, wrong codebase for verification).
- Logs of broken production runs:
  - `/tmp/qie_5513c_off.log` — 256²×20 with env unset, full NaN.
- F32 oracle artifacts:
  - `/home/ma-user/work/qie_f32_refs/` — per-op references from #201.
  - `/tmp/qie_q4_oracle_run.log` — Q4_0 dequant + matmul oracle from #202.
- Bundle backup: `/Users/yuechen/home/qie-saga-5.5.65-snapshot.bundle` on Mac.
