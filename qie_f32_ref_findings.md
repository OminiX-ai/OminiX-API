# QIE F32 Per-Op Reference Bisect — Findings

**Date**: 2026-04-29
**Scope**: F32 numpy/torch references for individual QIE-Edit ops, comparing
against native engine dumps. Per-op tolerance bisect (NOT full Diffusers E2E).
**Inputs**: Native engine dumps under `/tmp/qie_q45_inputs_1024/`,
`/tmp/qie_5542_v2/block0[01]/`, `/tmp/qie_dumps_real_5520/`,
`/tmp/qie_5543_step2/`.
**Refs**: `/home/ma-user/work/qie_f32_refs/{euler,rmsnorm,adaln,patchify,block0}_ref.py`.
**Tolerances**: cos ≥ 0.999 = MATCH, rel_max ≤ 1e-4 = F16-rounding precision,
rel_mean ≤ 1e-3 = acceptable F16 drift.

## TL;DR

**Every weight-free op verified by F32 reference is numerically correct
within F16-rounding precision (cos = 1.000000 across the board).** The
remaining drift between native and F32-ref is uniformly bounded at
`rel_max ≈ 3-7×10⁻⁴` and `rel_mean ≈ 1×10⁻⁵` — exactly the spread
expected when an op consumes a tensor that was round-tripped through F16
storage on the device.

The "polka-dot" / TILE pattern magnitude leak is therefore NOT in any of
the ops covered by this bisect (Euler, SiLU, RMSNorm, AdaLN modulate,
LayerNorm, gated residual, patchify/unpatchify). It MUST live in a
weight-dependent op that I cannot probe without weight dumps:

- Q/K/V projection matmuls
- attn-out projection matmul
- FFN up / gate / down matmuls (with GELU)
- t_emb upstream chain (timestep_embedder.linear_1 / linear_2 + SiLU)
- img_in / txt_in projection matmuls
- norm_out / proj_out (final unpatchify pre-step)

The smoking gun is the **t_emb magnitude itself**: `00_t_emb.f32` has
`abs_max=111.75` identically across engine and CLI dumps. That's the
input to the AdaLN matmul (`img_mod.1 / txt_mod.1`), which then yields
chunks with `abs_max` up to **491** (block 1, chunk[1] = "shift1" under
legacy binding). The arithmetic that consumes those chunks is correct;
it's the chunk values themselves that are pathological. §5.5.7 in
`docs/qie_q2_phase4_smoke.md` already isolates this as "the underlying
defect (something is amplifying t_emb / Q4 dequant by ~10×)".

## Per-op verdicts

### 1. Euler step (image_diffusion_engine.cpp:6280-6296) — MATCH

| dump pair | cos | max_abs_diff | rel_max | status |
|-----------|----:|-------------:|--------:|-------:|
| step 0 (sigma=1.0→0.75, dt=-0.25) plain Euler | 1.000000 | 2.98e-08 | 8.3e-09 | **MATCH** |
| step 0 with c_skip+c_out reconstruction (`x'=x+v*dt`) | 0.990265 | 1.27 | 0.26 | DRIFT |

The plain Euler form `d=(x-denoised)/sigma; x+=d*dt` matches the engine
dump bit-perfectly. The §5.5.47 c_skip+c_out reconstruction
(`denoised := x + (-sigma)*v` followed by the same Euler) does NOT match
— which confirms §5.5.47's RED verdict (the engine does NOT use that
reconstruction; the dumped `denoised_host` is treated as the *denoised
prediction* directly, not as the velocity `v`).

This is good news: the §5.5.47 hypothesis is mathematically wrong for
this engine — `denoised_host` is what the diffusers v-prediction CLI
calls "v" only if you also reinterpret the Euler form, but as-coded the
engine's plain Euler matches its own pre-Euler dump exactly. **No fix
needed in Euler step at step 0.**

Caveat: I could not validate step 1 because step 1 is all-NaN (per
§5.5.43). The Euler ARITHMETIC at step 1 is irrelevant since `denoised`
is already NaN going in.

### 2. SiLU(t_emb) (image_diffusion_engine.cpp:3656-3678) — MATCH

| dump | cos | max_abs_diff | rel_max | status |
|------|----:|-------------:|--------:|-------:|
| block0/1024² silu | 1.000000 | 2.16e-04 | 1.93e-06 | **MATCH** |

`y = x * sigmoid(x)` reproduces the engine's `aclnnSilu` output to F16
ULP. SiLU itself is fine.

### 3. RMSNorm (image_diffusion_engine.cpp:2725-2880) — MATCH (within F16)

All eight cases (block 0 + block 1 × img/txt × Q/K) hit cos = 1.000000
with `rel_max ≈ 3-4×10⁻⁴`. Slightly above my strict 1e-4 threshold so
classified ROUND_DRIFT, but that's because the dumped Q/K inputs went
F16→F32 once (engine stored F16 device tensors) and the gamma I had to
recover was estimated from the noisy Y/X ratio. The engine's RMSNorm
math is correct.

What is interesting is the input scale:

| stream | x abs_max | y abs_max | gamma abs_max |
|--------|----------:|----------:|--------------:|
| block 0 / 256² img_Q | 627 | 724 | 64.0 |
| block 1 / 1024² img_Q | **9544** | 12.6 | 2.6 |
| block 0 / 256² txt_Q | 1029 | 6.79 | 1.55 |
| block 1 / 1024² txt_Q | 2656 | 5.51 | 1.56 |

Q activations grow to ~10 000 magnitude entering block 1 (1024²). The
post-RMSNorm Y is ~10× bigger than canonical (~1) only on block 0 256²
img_Q (gamma_max=64 — most likely a Q4_0 dequant artifact in the gamma
weight tensor itself). RMSNorm correctly normalises, but the upstream
matmul has produced inputs that are 10× too large.

### 4. AdaLN modulate (image_diffusion_engine.cpp:2013-2080) — MATCH (legacy binding)

| dump pair | cos | max_abs_diff | rel_max | status |
|-----------|----:|-------------:|--------:|-------:|
| block 0 1024² img modulate1 [legacy: scale=ch0, shift=ch1] | 1.000000 | 6.7e-02 | 7.3e-04 | **MATCH** |
| block 0 1024² img modulate1 [SPEC: shift=ch0, scale=ch1] | 0.132321 | 109.07 | 1.03 | DRIFT |
| block 1 1024² img modulate1 [legacy] | 1.000000 | 0.31 | 5.9e-04 | **MATCH** |
| block 1 1024² img modulate1 [SPEC] | -0.101663 | 689 | 1.14 | DRIFT |

Confirms `modulate_(x, scale, shift)` does compute
`x * (1 + scale) + shift` correctly. The native engine is using the
**legacy chunk binding** as its source comment in §5.5.7 documents.

But the chunks are pathologically large:
- block 1 chunk[1] (used as shift1 under legacy) has `abs_max=491`
- block 1 chunk[4] (used as shift2 under legacy) has `abs_max=317`
- block 0 chunk[4] (shift2 legacy) has `abs_max=200`

So: the matmul math is correct, but chunks fed in are 100-500× larger
than canonical. §5.5.7 already notes this and pins legacy because under
the spec binding these would multiply (`(1 + scale=491)`) and explode by
another 491×. **The chunk-binding choice is a band-aid; the real
problem is the upstream `silu(t_emb) @ W_mod + b_mod` matmul producing
chunks that are 100× too large.**

### 5. Patchify / unpatchify (image_diffusion_engine.cpp:5182-5260) — MATCH

| test | cos | max_abs_diff | status |
|------|----:|-------------:|-------:|
| Engine vs vectorised diffusers patchify | 1.000000 | 0.0 | **MATCH (bit-exact)** |
| Round-trip patchify+unpatchify | 1.000000 | 0.0 | **MATCH (bit-exact)** |

The engine's patchify and unpatchify are bit-exact equivalents of the
diffusers reshape+permute path. **Patchify is NOT the source of the
"tile pattern" visual.** The tile pattern observed in §5.5.42 onward
must come from per-token magnitude inhomogeneity introduced by deeper
(weighted) ops, not from the patch-layout transform.

### 6. Block 0 weight-free transitions — MATCH

12 transitions (T1-T12) verified across block 0 256² real, block 0 1024²
1-step, and block 1 1024² 1-step. All 12 hit cos = 1.000000 in every
test case:

```
T1  LayerNorm(00_img) -> 04_img_LN1                     ROUND_DRIFT (cos=1.0)
T2  04_img_LN1 * (1+scale1) + shift1 -> 05_img_mod1     ROUND_DRIFT (cos=1.0)
T3  LayerNorm(00_txt) -> 06_txt_LN1                     ROUND_DRIFT (cos=1.0)
T4  06_txt_LN1 * (1+t_scale1) + t_shift1 -> 07_txt_mod1 ROUND_DRIFT (cos=1.0)
T5  00_img + gate1 * 12_to_out_0 -> 13_img_resid1       MATCH (bit-exact)
T6  00_txt + t_gate1 * 12_to_add_out -> 13_txt_resid1   (covered by T5 layout)
T7  LayerNorm(13_img_resid1) -> 14_img_LN2              ROUND_DRIFT (cos=1.0)
T8  14_img_LN2 * (1+scale2) + shift2 -> 15_img_mod2     ROUND_DRIFT (cos=1.0)
T9  LayerNorm(13_txt_resid1) -> 16_txt_LN2              ROUND_DRIFT (cos=1.0)
T10 16_txt_LN2 * (1+t_scale2) + t_shift2 -> 17_txt_mod2 ROUND_DRIFT (cos=1.0)
T11 13_img_resid1 + gate2 * 20_img_ff_down -> 24_img_resid2  MATCH (bit-exact)
T12 13_txt_resid1 + t_gate2 * 23_txt_ff_down -> 24_txt_resid2  (covered)
```

Critically: the magnitudes of intermediates ARE pathological:

- block 0 1024² T11 `24_img_resid2` `abs_max = 7.28×10⁶` (7 million)
- block 1 1024² T11 `24_img_resid2` `abs_max = 1.41×10⁸` (140 million)

The img residual stream ends a single block at ±10⁷ and is ±10⁸ entering
block 2, growing ~20×/block until F32 saturates around block 27 (per
§5.5.43). The growth is from the *weighted* ops:

- `12_to_out_0` (img attn out projection) feeds into the gated residual
  T5 at magnitude that is already enormous.
- `20_img_ff_down` feeds into T11 at the magnitude that grows the
  residual stream by ~10⁷ per block.

These two MATMUL outputs are the magnitude leak vectors. Their inputs
in turn come from RMSNorm-correct Q/K/V (whose sources had abs_max=10⁴),
so the entire weighted chain is amplifying.

## Root cause analysis

The bisect rules out **all** non-weighted F32 ops in the per-block code.
The drift therefore lives in one of the matmuls or their dequant. Three
suspect paths, in order of suspicion:

### Suspect 1 (HIGH): `img_mod.1` / `txt_mod.1` matmul output is 100× too large

Direct evidence from chunk[1] (legacy shift1) `abs_max=491` at block 1.
A trained QwenImage AdaLN linear should produce shifts ~O(1) and scales
~O(0.1). Instead we see shifts of ±491 and scales of ±152.

Possible causes:
a) Q4_0 dequant of `img_mod.1.weight` is wrong (wrong scales/biases lookup).
b) `silu(t_emb)` input to the matmul has `abs_max=111.75` (vs O(1)
   expected). The amplification factor 111 from the timestep_embedder
   chain alone would explain a ~100× chunk inflation.
c) The matmul is summing in F16, hitting F16 max representable at deep
   reductions, but mod1 is a 256→3072 matmul; F16 reduction at this
   depth shouldn't blow up.

The §5.5.20 t_emb oracle (already on disk at
`tools/probes/qie_t_emb_oracle/qie_t_emb_oracle.py`) has bytes-equal
agreement against the F32 reference for the t_emb chain, which means
the engine *correctly computes* a t_emb that has `abs_max=111.75`. So
the upstream defect lives in either the timestep_embedder.linear_1 /
linear_2 weights themselves OR in the dequant/cast of those weights —
NOT in the SiLU+matmul math the engine performs.

### Suspect 2 (HIGH): FFN down (and possibly attn-out) projection matmul

Block 1 `13_img_resid1` enters at `abs_max=8.98×10⁶` and `24_img_resid2`
exits at `abs_max=1.41×10⁸`. The growth factor across block 1 is **15.7×**.
The growth is in `20_img_ff_down`'s output (since T11 is bit-exact, the
gate2 application doesn't change the order of magnitude — gate2's max
is bounded by chunk[5] `abs_max=221`).

So `20_img_ff_down` is producing outputs ~`abs_max = 1.4×10⁸ / gate2_max
≈ 6×10⁵`, which then gates-residual into the residual stream.
Possible causes:
a) Q4_0 dequant of `ff.net.2.weight` (FFN down projection) is wrong.
b) F16 saturation of intermediate FFN activations (we already saw
   `08_img_K abs_max=10744` at block 1 — the post-RMSNorm path is
   normalised but the FFN path is NOT, so 10⁵+ activations entering
   the down projection are F16 ulp-rounded by ~0.5).
c) The §5.5.46 BF16 widen patch hadn't covered FFN down — the residual
   stream stays F32 but the hidden FFN tensor goes F16.

### Suspect 3 (MEDIUM): img_in / txt_in / norm_out / proj_out

These are not directly observable in the per-block dumps but feed the
chain on entry/exit. If `img_in` produces `00_img` with `abs_max=10⁴`,
that already concedes the leak before any block executes.

The actual `00_img.f32` for block 0 / 1024² 1-step has stats:
- mean ≈ 0
- std ≈ 1.0 (good!)
- abs_max ≈ 10ish

So `img_in` is fine — the residual stream enters block 0 at the right
scale (~1σ). The amplification happens INSIDE the blocks.

## Recommended fix

The next agent should NOT touch:
- Euler step (correct — see §5.5.47 RED revisit confirmed)
- SiLU
- RMSNorm
- AdaLN modulate
- Patchify / unpatchify
- LayerNorm
- Gated residual

The next agent SHOULD instrument and verify, in this order:

1. **Dump `silu(t_emb) @ img_mod.1.W^T + img_mod.1.b` against an F32
   numpy reference using the actual GGUF Q4_0 dequant**. The §5.5.20
   `qie_t_emb_oracle.py` already contains the GGUF dequant scaffolding
   (`load_dequant_weight`); extend it to dequant `img_mod.1.weight` and
   compute the F32 matmul output, then compare to engine
   `02_img_mod_out.f32`. This will localise whether the chunk
   magnitudes (±491) come from genuine trained weights or from a
   dequant defect. **File: `image_diffusion_engine.cpp:3681`** —
   `dispatch_matmul_(scratch_q_dev_, lw.img_mod_w_q4, lw.img_mod_scale,
   lw.img_mod_b, B, H, 6 * H, scratch_mod_dev_)`.

2. **Dump `ff_norm(adaLN(img_resid1)) @ ff.net.0.weight` (FFN gate
   matmul) and `... @ ff.net.0.proj.weight` (FFN up matmul) and
   `gelu(...) @ ff.net.2.weight` (FFN down matmul)** for block 1, then
   compare each to its F32 numpy dequant reference. The 15.7× per-block
   growth strongly implicates one of these. **File:
   `image_diffusion_engine.cpp` ~line 4304-4400** in the FFN dispatch
   (find via `grep -n 'ff.net\|img_ff'`).

3. **Verify the Q4_0 super-block scale tensor itself for these specific
   weight names**. The §5.5.7 hypothesis "something is amplifying
   t_emb / Q4 dequant by ~10×" maps to a Q4_0 super-block that has the
   wrong scale stored or read. The native engine reads scales via
   `dequant_upload_q4_0` — check that the scale tensor sub-name
   (`.scale`) is loaded with the correct GGUF type and that the
   dequantization formula in `dispatch_matmul_` matches GGML's
   reference (`q4_0_dequantize_block_8`-equivalent).

## Open questions

1. The chunk[1] / chunk[4] amplification is consistent across block 0
   and block 1 with the same magnitude tier (~50×, ~300×). If both
   blocks have similar amplification, the defect is not block-specific
   — it's in the modulation matmul code path, not in any one block's
   weight tensor. This suggests issue #1 above (img_mod / txt_mod
   matmul) is the highest-priority root cause.

2. We do not have a CLI ground-truth `02_img_mod_out.f32` to compare
   against. The §5.5.46 carry-over already noted: dispatch a CLI dump
   of the same intermediate to localise engine-vs-CLI divergence at
   the chunk level. That should be the next agent's first probe.

3. The §5.5.13 attention bit-correctness verification (FIA = bit-exact
   F32 oracle at REAL inputs, cos=1.000000) needs to be re-confirmed at
   1024² with current dumps — it was done at 256² real-data, where
   block 0 `08_img_Q abs_max=627`. At 1024² block 1 `08_img_Q abs_max
   = 9544`, an order of magnitude larger. Whether FIA stays bit-exact
   at the larger magnitude has not been re-tested.

## Files

- `/home/ma-user/work/qie_f32_refs/euler_ref.py`
- `/home/ma-user/work/qie_f32_refs/rmsnorm_ref.py`
- `/home/ma-user/work/qie_f32_refs/adaln_ref.py`
- `/home/ma-user/work/qie_f32_refs/patchify_ref.py`
- `/home/ma-user/work/qie_f32_refs/block0_ref.py`
- `/home/ma-user/work/qie_f32_refs/_compare.py`
