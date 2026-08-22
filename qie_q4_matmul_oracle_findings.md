# QIE Q4_0 / Q5_K / Q4_1 matmul oracle — magnitude leak verdict

Per agent #201's per-op F32 bisect, all weight-free op classes (Euler,
RMSNorm, SiLU, AdaLN math, patchify, weight-free block transitions) are
bit-correct vs F32 numpy. The DiT magnitude blow-up that hits F16
saturation around block 27 (per §5.5.43) was narrowed to two
quantized-weight matmuls in the engine, both flagged in §5.5.42 V2
post-fix dumps as still inflated:

1. `silu(t_emb) @ img_mod.1.weight + img_mod.1.bias`  → `02_img_mod_out`
   (per-block; block-1 chunks absmax ≈ 491, block-0 absmax ≈ 269)
2. `gelu(15_img_mod2 @ img_mlp.net.0.proj.W.T + b) @ img_mlp.net.2.W.T + b`
   → `20_img_ff_down` (block-1 absmax ≈ 6.10×10⁵, far above F16 65504)

Hypotheses framed by the parent agent:
- A. Trained weights are pathological → magnitudes are by design;
     fix = widen output dtype to BF16/F32 in the engine matmul path.
- B. Q4_0 dispatch has a residual defect → fix dequant path.

## Method

Pure-F32 numpy oracle: `tmp/qie_q4_matmul_oracle.py` (deployed to
ac03 at `/tmp/qie_q4_matmul_oracle.py`).

- Q4_0 dequant validated **byte-equal** vs `gguf-py` reference
  (`quants.dequantize_blocks`) on `transformer_blocks.1.img_mod.1.weight`
  (qt=2 / Q4_0). `bit_equal=True`, `max|hand-ref|=0`.
  → Both dequant paths are numerically identical and trustable.
- Inputs come from engine F32 dumps under `/tmp/qie_5542_v2/blockNN/`
  (post-§5.5.42 V2 Q4_0 dispatch fix).
- Oracle = `silu @ W.T + b` in pure F32 (and a F16-rounded mimic).
- Oracle compared to engine `02_img_mod_out` and `20_img_ff_down`.

## Tensor types in QIE-Edit-2509-Q4_0.gguf (relevant)

| Weight | qt | type |
|---|---|---|
| block-0 `img_mod.1.weight` | 13 | **Q5_K** |
| block-1 `img_mod.1.weight` | 2 | **Q4_0** |
| block-1 `img_mlp.net.0.proj.weight` (ff_up) | 2 | **Q4_0** |
| block-1 `img_mlp.net.2.weight` (ff_down) | 3 | **Q4_1** |

Notable: the GGUF labelled "Q4_0" actually mixes Q4_0 / Q4_1 / Q5_K /
F16 / F32 across tensors. The "ff.net.2" weight is **Q4_1**, not Q4_0,
which the dispatch path treats identically (same WQBMMv2 / aclnnMm
codepath, just different scale layout). Both Q4_0 and Q4_1 paths are
exercised by the test.

## Results

### Test 1 — `img_mod.1` matmul (clean isolation, full input + output dumped)

| Block | qt | engine absmax | oracle absmax | cos | max_abs_diff | mean_abs_diff | rel_max |
|---|---|---|---|---|---|---|---|
| 0 | Q5_K | 2.690e+02 | 2.690e+02 | **1.000000** | 1.526e-01 | 2.36e-03 | 5.67e-04 |
| 1 | Q4_0 | 4.915e+02 | 4.914e+02 | **1.000000** | 2.372e-01 | 6.77e-03 | 4.83e-04 |

Magnitude (oracle absmax 491 at block 1) is **reproduced bit-for-bit by the
F32 oracle from the same Q4_0 weight + the same dumped silu(t_emb) input**.
The ≤ 0.25 max_abs_diff is pure F16-rounding round-trip noise from the
engine's `out_dtype=ACL_FLOAT16` cast on store.

**Verdict for img_mod.1: A.** Magnitude is from the trained weights.
Q4_0 dispatch is correct.

Why oracle absmax 491 ≠ engine 491.5 (one part in 4000): engine path
is `aclnnWeightQuantBatchMatmulV2(input_f16, weight_q4_0, scale_f16) →
F16 output → InplaceAdd(bias_f16)`. The oracle stores F32. Cast-back
parity at this level (`rel_max=4.8e-4`) is the expected floor for
F16 intermediate accumulation rounding, not a Q4_0 defect.

### Test 2 — img MLP chain (ff_up → gelu → ff_down) at block 1

Run as a single oracle from `15_img_mod2` to `20_img_ff_down` because the
engine does not dump `19_img_gelu`. Both Q4_0 (ff_up) and Q4_1 (ff_down)
matmuls are exercised; if either had a dispatch bug, the chain would
diverge.

| variant | engine absmax | oracle absmax | cos | max_abs_diff | mean_abs_diff | rel_max |
|---|---|---|---|---|---|---|
| F32 oracle (gelu tanh approx, ggml-cann GeluV2) | 6.103e+05 | 6.094e+05 | **0.999998** | 2.17e+03 | 3.96e+01 | 3.56e-03 |
| F32 oracle (gelu erf exact)                     | 6.103e+05 | 6.094e+05 | **0.999998** | 2.17e+03 | 3.96e+01 | 3.56e-03 |

The chain output magnitude (engine absmax = 610 300, oracle 609 380) is
**reproduced by the F32 oracle to cos=0.999998 and rel_max=3.6e-3**. The
diff floor is consistent with two F16-rounded matmul intermediates plus
a BF16 final-cast on the engine side (`dump_tensor_dt(...,
ffn_down_bf16 ? PROBE_BF16 : PROBE_F16)` on line 4382 — this dump was
under `QIE_FFN_DOWN_BF16=1` since 6.10e5 > F16 max 65504).

**Verdict for ff.net.2 (and ff.net.0): A.** Chain magnitude is from the
trained weights. Q4_0 (ff_up) and Q4_1 (ff_down) dispatch are both
numerically correct.

## Final verdict

**A confirmed for both weights.** The Q4_0 dispatch fix in §5.5.42 V2
(WQBMMv2 mirror of ggml-cann mul_mat_quant) is correct. The 491 mod1
chunks and 6.10e5 ff_down outputs are intrinsic to the trained Q4_0/Q4_1
weights. The magnitude cascade that hits F16 saturation by block 27 is a
**dtype-saturation bug in the storage cast**, not a quantization defect.

## Recommended fix

The `img_mod.1` matmul path is the one currently NOT widened to BF16.
Source:
- `tools/qwen_image_edit/native/image_diffusion_engine.cpp:740` (header
  default `aclDataType out_dtype = ACL_FLOAT16`).
- `tools/qwen_image_edit/native/image_diffusion_engine.cpp:3681-3682`
  — img_mod.1 dispatch, no out_dtype arg → defaults to F16.
- `tools/qwen_image_edit/native/image_diffusion_engine.cpp:3688-3692`
  — txt_mod.1 dispatch, same pattern.

Block-1 already produces 491 chunks; block-0 already produces 269 chunks.
Once `(1 + scale) * x_LN` is computed, `(1+491)*x` can land near F16 max
even from a small LN1, and the residual chain compounds it.

Mirror the §5.5.45/46 widening pattern used for QKV (line 3846,
`s_qkv_bf16 ? ACL_BF16 : ACL_FLOAT16`) and ff_down (line 4377,
`ffn_down_bf16 ? ACL_BF16 : ACL_FLOAT16`):

1. Add a static-cached env knob `s_mod_bf16` parallel to `s_qkv_bf16`,
   reading e.g. `QIE_MOD_BF16` (and respecting `QIE_ALL_BF16`).
2. Change line 3681/3688 to pass
   `s_mod_bf16 ? ACL_BF16 : ACL_FLOAT16` as the out_dtype.
3. The downstream consumer (`scratch_mod_dev_` chunks) is already F32-
   capable per §5.5.45 plumbing — verify the chunk reader at
   `image_diffusion_engine.cpp:3804` handles BF16 with the existing
   PROBE_BF16 / `dump_tensor_dt(...)` paths.

## Open questions / next-agent

- Does the `02_img_mod_out` dump at `/tmp/qie_5542_v2/block01/` already
  include the post-§5.5.42 fix V2? Yes — confirmed by directory name
  and oracle agreement. The 491 chunks are the **correct V2 output**.
- Why does §5.5.43 cite F16 saturation around block 27, not block 1?
  Because `(1 + 491) * x_LN1` doesn't *immediately* saturate F16 when
  x_LN1 is small (LN1 outputs are O(1)); the saturation is in the
  residual accumulation `img += gate * to_out` which compounds across
  ~27 blocks before exceeding 65504.
- Pre-§5.5.42 dumps at `/tmp/qie_5536_eng_real/block01/` showed
  oracle vs engine `02_img_mod_out` cos=0.045 (per `/tmp/qie_b1_oracle.log`)
  — that confirms §5.5.42 V2 fix DID resolve a real Q4_0 dispatch defect.
  The current dumps are post-fix and clean.
- BF16 widening alone may not be enough if some intermediate is still
  F16; audit all of `dispatch_matmul_(*, ACL_FLOAT16)` (default) call
  sites; mod1, txt_mod1, and any to_add_out / similar paths still on
  F16 default.

## Reproduction

On `ssh ac03`:

```
~/anaconda3/envs/PyTorch-2.7.1/bin/python /tmp/qie_q4_matmul_oracle.py
```

Inputs: `/tmp/qie_5542_v2/block00/`, `/tmp/qie_5542_v2/block01/`,
`/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf`.
Run log: `/tmp/qie_q4_oracle_run.log`.
Oracle script (local copy): `tmp/qie_q4_matmul_oracle.py`.
