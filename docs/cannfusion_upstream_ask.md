# CannFusion Feature Request: A16W8 (F16×INT8→F16) Dtype Lane

**Project evaluating**: OminiX-API / Qwen3-TTS production inference on Ascend 910B4
**CannFusion rev examined**: `539fc01` on `main`, v0.2.0 release
**Date**: 2026-04-21
**Context**: `§1.12` is resolved in-tree — we confirmed the fused-epilogue
cube+vec pipeline now dispatches cleanly on CANN 8.3.RC1 / Ascend 910B4
with zero mismatches. Huge win, thanks. We wanted to adopt CannFusion for
the FFN sublayer of our Qwen3-TTS code-predictor hot path, but the
remaining blocker is a **dtype-whitelist mismatch**.

## What we need

The production dtype used by our Qwen3-TTS engine (and by essentially
every on-device LLM/TTS stack using weight-only quantisation today) is:

- **A** (activation): `F16`
- **B** (weight):     `INT8`, per-channel symmetric quant
- **scale**:          `F16` per-channel, shape `[N]`
- **acc** (internal): `F32` (Cube Unit accumulator, standard)
- **Y** (output):     `F16`

Concretely: Huawei's own `aclnnWeightQuantBatchMatmulV3` takes exactly
this shape, and it's the op our CP engine dispatches today. We'd like to
replace its tail ops (SiLU, mul, residual-add) with your epilogue fusion
for 5 calls/layer × 5 layers × 15 forwards/frame in our hot path.

## Why this combo specifically

"Weight-only" int8 quant (A16W8) is the common sweet spot for small
on-device LLM/TTS decoding:

- Activations stay F16 to preserve numerical sensitivity in
  norm/softmax-adjacent ops.
- Weights go INT8 to halve weight RAM (2.7 GB → 1.5 GB for our model)
  and halve HBM bandwidth on the weight leg.
- Output stays F16 to feed directly into subsequent F16 ops.

It's not a research corner case — it's the majority of production
int8 inference on Ascend right now.

## What's missing in CannFusion

We tried to generate this combo with a minimal TOML:

```toml
[matmul]
a_type = "f16"
b_type = "int8"
acc_type = "f32"
out_type = "f16"
# ... tile 128×128×128, epilogue [dequant, cast]
```

Validator response:
```
Caused by:
    0: validation failed before codegen: unsupported dtype combination:
       A=F16 B=INT8 acc=F32 out=F16 (Cube Unit on 910B does not implement this combo)
    1: unsupported dtype combination: A=F16 B=INT8 acc=F32 out=F16 ...
```

Looking at `src/validate.rs:141-163`, the whitelist is hard-coded:
```rust
matches!(
    combo,
    (F16,F16,F16) | (F16,F16,F32) | (BF16,BF16,F32) | (F32,F32,F32)
  | (INT8,INT8,INT32) | (INT8,INT8,F32)
)
```

And there's an explicit negative unit test at `validate.rs:367-372`
(`dtype_rejects_f16_int8_f32`), so the rejection is intentional design,
not an omission.

The Cube Unit on 910B absolutely supports mixed-precision MAC via the
L0B-stage dequant pattern (bisheng + AscendC primitives permit it —
that's how `aclnnWeightQuantBatchMatmulV3` is implemented), so the
hardware constraint cited in the error text is about your codegen
choices, not a silicon limit.

## What an A16W8 lane would require (codebase delta estimate)

Based on reading your tree at `539fc01`, roughly:

1. **Validator** (`src/validate.rs`): extend the whitelist + add
   positive test. ~10 lines.
2. **Tiling key** (`src/codegen/context.rs`): the current TILING_KEY
   dispatch is `1=f32 2=f16 3=bf16 4=int8` — single-dtype. Needs a
   mixed-precision key, probably encoding `(A_dtype, B_dtype, out_dtype)`.
3. **Tiling data** (`src/tiling.rs` + generated `tiling_data.h`): add
   per-channel scale tensor metadata (pointer + shape + dtype). Optional
   zero-point for asymmetric quant.
4. **Kernel template** (`templates/kernel.h.tera`): emit an L0B-stage
   dequant before the Cube commit. AscendC has `DataCopy` + `Cast` + a
   `Muls` against a broadcast scale tile — pattern is similar to your
   existing `bias` epilogue in the vec kernel, just moved to the B-leg.
5. **Host API** (`templates/api.cpp.tera`): new signature accepting
   the scale tensor handle.
6. **Fixtures** (`tests/fixtures/codegen/`): one or two A16W8 TOMLs,
   referenced from `tests/fixtures/valid_basic.toml` style.
7. **Device-smoke reference**: host-side compute for A16W8 parity check,
   similar to your existing `m12e_cube_bias_silu_cast.toml` runner.

Rough estimate: 2-4 days for a prototype, probably 1-2 weeks through
full CI + autotune + perf baseline against `aclnnWeightQuantBatchMatmulV3`.

## Performance upside we'd offer back

If this lane lands, our intent is to ship a fused
`down-Mm(A16W8) + residual-add` kernel for Qwen3-TTS CP and measure
real-NPU wall against the aclnn chain. Device baselines from our probe:

- CP forward stock (A16W8 via aclnn): ~1.54 ms/forward on 910B4
- Each forward has 7 W8 matmuls + 3 elementwise (SiLU/mul/add) = ~10
  aclnn calls
- Every fused Mm+epilogue kernel replacing a 2-op chain saves
  ~0.05 ms/call × 5 layers = ~0.25 ms/forward ≈ 5-8% fps

That's a real public benchmark number on the exact
`cp_hidden=1024, inter=3072, head_dim=128` shape class, with a
numeric-parity gate. We'd contribute the benchmark back as a
`tests/fixtures/` entry.

## What's already verified

From our F0 / F-re probes:
- CannFusion 0.2.0 builds cleanly via `cargo build --release`.
- Fresh generation works for F16×F16 fused-epilogue (`m12e_cube_bias_silu_cast.toml`).
- Compile on ac01 / ac03 / CANN 8.3.RC1 via bisheng/ccec/lld with the
  project's own `scripts/device-smoke.sh` — zero mismatches, runner PASS.
- §1.12 fault (`ACL_ERROR_RT_AICORE_EXCEPTION 507015`) is gone.

Toolchain is not the blocker; the blocker is purely the dtype whitelist.

## Our ask

Is A16W8 (F16×INT8→F16 with per-channel F16 scale) on the roadmap in
any form? If not, and you'd be open to it, we'd be happy to:

1. Sponsor the design with real-world shape/autotune data.
2. Run device-smoke parity on ac03 (CANN 8.3.RC1 / 910B4) against the
   benchmarks as you iterate.
3. Contribute the Qwen3-TTS CP benchmark fixture once it lands.

Let us know what would be most useful. Happy to open a GitCode issue
with this content if that routes better for triage.
