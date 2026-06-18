# QIE_RESID_CLAMP disambiguation — block-0 `13_img_resid1` divergence

**Task**: Confirm whether the `QIE_RESID_CLAMP=60000` band-aid (added §5.5.44) is masking the actual residual divergence at block 0 step 0, or whether the divergence is upstream (genuine substep bug).

**Result**: Clamp is **irrelevant** to the block-0 `13_img_resid1` divergence. The cos=0.0507 / mag_ratio=0.491 finding stands without any clamp interference. The substep bisect on ac03 (#207) is the right path.

---

## Method

Originally planned: rebuild test binary on ac01, run with `QIE_RESID_CLAMP=0`, compare dumped `13_img_resid1.f32` to CLAMP=60000 baseline.

Actual approach: source-level proof by code inspection (cheaper, exact). Empirical run on ac01 was prepared but not executed because the proof is conclusive without it.

## Source-level proof

Engine source (`tools/qwen_image_edit/native/image_diffusion_engine.cpp`, ac03 HEAD `3daae48`, lines 4310-4334):

```cpp
// Step 8: gated residual add — img += attn_out * gate1
gated_residual_add_f32_(img_hidden, ...);
gated_residual_add_f32_(txt_hidden, ...);

// 13_img_resid1 dump fires HERE — line 4316
dump_tensor_f32("13_img_resid1.f32", img_hidden, ...);
dump_tensor_f32("13_txt_resid1.f32", txt_hidden, ...);

// QIE_RESID_CLAMP block fires HERE — line 4322 (AFTER dump)
{
    static float s_clamp = -1.0f;
    if (s_clamp < 0.0f) {
        const char *v = std::getenv("QIE_RESID_CLAMP");
        s_clamp = (v && *v) ? (float)std::atof(v) : 60000.0f;
    }
    if (s_clamp > 0.0f) {
        clamp_residual_f32_(img_hidden, ...);
        clamp_residual_f32_(txt_hidden, ...);
    }
}
```

**Key observation**: `dump_tensor_f32("13_img_resid1.f32", ...)` (line 4316) runs **before** the `QIE_RESID_CLAMP` block (line 4322). The dump captures the F32 residual in its **pre-clamp** state.

### Single-fire dump guard

Lines 3514-3519:
```cpp
static std::set<int> s_dump_fired;
const bool match_multi = !s_dump_dir.empty() && !s_dump_indices.empty()
                          && s_dump_indices.count(s_block_idx_now) > 0
                          && s_dump_fired.count(s_block_idx_now) == 0;
if (match_multi) s_dump_fired.insert(s_block_idx_now);
```

`s_dump_fired` is a process-lifetime static. Each block index dumps exactly **once** — on the first call to `forward_block_(block_idx)`. For block 0, that is step 0.

### No prior clamp can have run

At block 0 step 0:
- `s_intra_calls` transitions 0 → 1 (this is the first `forward_block_` invocation)
- No prior block call exists in the process
- Therefore no prior `clamp_residual_f32_` call has ever executed
- The clamp value setting (60000 or 0) cannot influence any input to block 0 step 0

## Magnitude check on ac03 baselines

```
eng_baseline (CLAMP=60000):  N=1572864 mean=-2.255e-01 std=6.202e+01
                              absmax=1.235e+03 nan=0 inf=0
cli_baseline:                N=1572864 mean= 2.712e+00 std=1.702e+02
                              absmax=2.516e+03 nan=0 inf=0
cos(eng60000, cli) = 0.05068731
mag_ratio absmax eng/cli = 0.49089465
```

Engine absmax = **1234.9**, CLI absmax = **2515.6**. Both are **far below the 60000 clamp threshold** — even if the clamp had run before this dump (it didn't), it would have been a no-op on these tensors. The clamp couldn't possibly trigger at block 0 step 0 with these magnitudes.

The 3.26e10 magnitude mentioned in the task context refers to **deep blocks (block 27+) at later steps**, not block 0 step 0. At block 0 step 0, the residual is well-conditioned (~1000-2500 absmax) and the divergence is NOT a saturation issue.

## ac01 setup state (for reference)

- `OminiX-Ascend-w1` synced from ac03 main via git bundle: now at `3daae48`.
- ac01 NPU (910B4, 32GB HBM) idle and healthy.
- Inputs synced: `/tmp/qie_q45_inputs/` (3.2 MB, all 5 .f32.bin tensors).
- Baselines synced: `/tmp/qie_5513f_eng_blocks/block00/13_img_resid1.f32` and `/tmp/qie_5513f_cli_blocks/block00/qie_cli_blk00_13_img_resid1.f32.bin`.
- GGUF (12 GB) was not transferred — bridging through local mac would have taken ~30 min and the source-level proof made it unnecessary.

If the empirical run is still wanted, the only blocker on ac01 is the GGUF transfer; everything else is staged.

## Verdict

**Clamp is IRRELEVANT to the block-0 `13_img_resid1` divergence** (NOT a band-aid masking it).

Reasons:
1. Dump occurs strictly before clamp in source order — clamp setting cannot affect this dump.
2. Single-fire `s_dump_fired` guard means block-0 dump is captured on the very first `forward_block_` call (step 0), before any clamp has ever executed.
3. Tensor magnitudes at block 0 step 0 (absmax ~1235 engine, ~2516 CLI) are an order of magnitude below the 60000 clamp threshold, so even if the clamp had run it would have been a no-op.
4. Empirical CLAMP=0 vs CLAMP=60000 dumps would be **bitwise identical** — not a useful diagnostic at block 0.

### Implication for #207

The cos=0.0507 / mag_ratio=0.491 divergence at block-0 `13_img_resid1` is **genuine, upstream, and clamp-independent**. The substep bisect on ac03 (#207) is the correct diagnostic path — the bug is in one of the substeps that produce `img_hidden` after the gated residual add: img/txt LN1, mod1, attention QKV, RMSNorm, RoPE, attn_out, or the gated residual add itself. The clamp is a separate concern for deep blocks at later steps where the residual stream grows large.

The §5.5.44 clamp band-aid is a downstream-block / late-step hack for F16 LN saturation — it has no bearing on the block-0 divergence under investigation.
