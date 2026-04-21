First: thank you for resolving §1.12 — we confirmed the fused-epilogue cube+vec pipeline now dispatches cleanly on CANN 8.3.RC1 / Ascend 910B4 with zero mismatches via your own `scripts/device-smoke.sh`. Huge win.

We've been evaluating CannFusion for the FFN sublayer of a production Qwen3-TTS code-predictor hot path on 910B4. The remaining blocker for us is a **dtype-whitelist mismatch**, and we wanted to write this up carefully in case it fits your roadmap.

## What we need

The dtype our engine uses (and the one most on-device LLM/TTS stacks using weight-only quantisation use) is:

- **A** (activation): `F16`
- **B** (weight):     `INT8`, per-channel symmetric quant
- **scale**:          `F16` per-channel, shape `[N]`
- **acc** (internal): `F32` (Cube Unit accumulator)
- **Y** (output):     `F16`

Huawei's own `aclnnWeightQuantBatchMatmulV3` takes exactly this shape, and it's the op our CP engine dispatches today.

## Why this combo specifically

"Weight-only" int8 quant (A16W8) is the standard pattern for on-device decoding:

- Activations stay F16 to preserve numerical sensitivity in norm/softmax-adjacent ops.
- Weights go INT8 to halve weight RAM (2.7 GB → 1.5 GB for our model) and halve HBM bandwidth on the weight leg.
- Output stays F16 to feed directly into subsequent F16 ops.

It's not a research corner case — it's the majority of production int8 inference on Ascend right now.

## What we tried

Minimal TOML:

```toml
[matmul]
a_type = "f16"
b_type = "int8"
acc_type = "f32"
out_type = "f16"
# tile 128×128×128, epilogue [dequant, cast]
```

Validator response:
```
validation failed before codegen: unsupported dtype combination:
   A=F16 B=INT8 acc=F32 out=F16 (Cube Unit on 910B does not implement this combo)
```

Looking at `src/validate.rs:141-163`, the whitelist is hard-coded:
```rust
matches!(
    combo,
    (F16,F16,F16) | (F16,F16,F32) | (BF16,BF16,F32) | (F32,F32,F32)
  | (INT8,INT8,INT32) | (INT8,INT8,F32)
)
```

And there's an explicit negative unit test at `src/validate.rs:367-372` (`dtype_rejects_f16_int8_f32`), so the rejection looks like intentional design.

The Cube Unit on 910B does support mixed-precision MAC via the L0B-stage dequant pattern (that's how `aclnnWeightQuantBatchMatmulV3` is implemented) — so the hardware constraint cited in the error text is, we think, about codegen scope rather than a silicon limit. Happy to be corrected if we're wrong about that.

## Rough codebase delta (if this were to land)

Based on reading the tree at `539fc01`:

1. **Validator** (`src/validate.rs`): extend whitelist + positive test. ~10 lines.
2. **Tiling key** (`src/codegen/context.rs`): current `TILING_KEY` is single-dtype (`1=f32 2=f16 3=bf16 4=int8`). Needs a mixed-precision key.
3. **Tiling data** (`src/tiling.rs` + generated `tiling_data.h`): add per-channel scale tensor metadata.
4. **Kernel template** (`templates/kernel.h.tera`): emit an L0B-stage dequant (`DataCopy` + `Cast` + `Muls` against a broadcast scale tile) before the Cube commit.
5. **Host API** (`templates/api.cpp.tera`): new signature accepting the scale tensor handle.
6. **Fixtures** (`tests/fixtures/codegen/`): one or two A16W8 TOMLs.
7. **Device-smoke reference**: host-side compute for A16W8 parity check.

Rough estimate: 2-4 days prototype, 1-2 weeks through full CI + autotune + perf baseline.

## What we'd contribute back

If an A16W8 lane lands, our intent is to ship a fused `down-Mm(A16W8) + residual-add` kernel for Qwen3-TTS CP and measure real-NPU wall against the aclnn chain. Device baselines we've already taken on 910B4:

- CP forward stock (A16W8 via aclnn): ~1.54 ms/forward
- Per forward: 7 W8 matmuls + 3 elementwise (SiLU/mul/add) ≈ 10 aclnn calls
- Each fused Mm+epilogue substitution saves ~0.05 ms/call × 5 layers ≈ 0.25 ms/forward ≈ 5-8% fps

That's a concrete benchmark number on the `cp_hidden=1024, inter=3072, head_dim=128` shape class with a numeric-parity gate. We'd contribute the benchmark back as a `tests/fixtures/` entry, plus any quality gates you'd want.

## What's already verified on our side

From local F0 / F-re probes on CANN 8.3.RC1 + 910B4:

- CannFusion 0.2.0 builds cleanly via `cargo build --release`.
- F16×F16 fused-epilogue generation works (`m12e_cube_bias_silu_cast.toml`, 128×128×128).
- Compile via bisheng/ccec/lld + `scripts/device-smoke.sh` reports `0 mismatches, max_abs=0.0078125`, runner PASS.
- §1.12 fault (`ACL_ERROR_RT_AICORE_EXCEPTION 507015`) is gone.

Toolchain / §1.12 are not the blocker; it's purely the dtype whitelist.

## Our ask

Is A16W8 (F16×INT8→F16 with per-channel F16 scale) on the roadmap in any form? If not, and you'd be open to it, we'd be happy to:

1. Sponsor the design with real-world shape/autotune data.
2. Run device-smoke parity on 910B4 / CANN 8.3.RC1 against the iterations as you go.
3. Contribute the Qwen3-TTS CP benchmark fixture once it lands.

Happy to discuss in this thread or move to email if that fits better. Thanks again for the §1.12 work.
