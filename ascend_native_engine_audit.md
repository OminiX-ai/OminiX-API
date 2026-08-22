# Ascend Native QIE-Edit Engine — Readiness Audit

**Auditor:** Claude (Opus 4.7, 1M context)
**Date:** 2026-04-29
**Scope:** Read-only audit of `tools/qwen_image_edit/native/image_diffusion_engine.{h,cpp}` to decide if the native engine is production-ready, partial, or dead code.
**Sources read:**
- `/Users/yuechen/home/OminiX-Ascend/tools/qwen_image_edit/native/image_diffusion_engine.h` (870 LOC)
- `/Users/yuechen/home/OminiX-Ascend/tools/qwen_image_edit/native/image_diffusion_engine.cpp` (5,647 LOC)
- `/Users/yuechen/home/OminiX-Ascend/docs/qie_q2_phase4_smoke.md` (3,232 LOC saga)
- ac03 git log + `/tmp/qie_q45_step4*` artefact inventory + ac03 build dir

---

## Verdict at a glance

**Branch B — engine is REAL and FUNCTIONAL but NOT visually correct yet.**

- The `init_from_gguf` → 60-block `forward_block_` → `denoise_full` (Phase 4.5 Step 4) → host-side patchify/unpatchify → flow-Euler chain is fully implemented and runs end-to-end on ac03 against the real `Qwen-Image-Edit-2509-Q4_0.gguf`.
- A 20-step run at 32×32 latent (256² eye-check resolution) produces a finite latent with NaN=0, std=4.86, mean=-2.46 — **GREEN on the numerical gate** at ~1.17 s/step (~24 s wall).
- The eye-check PNG is finite + structured but **shows a coherent tile pattern, not a recognizable cat** — the engine's attention path or unpatchify-host code still has a residual numerical/layout bug that the §5.5.x bisect has been narrowing.
- The **public `denoise()` (header line 362, body line 1264)** is still a Phase-2 stub returning false. Production runs go through `denoise_full()` (header line 510, body line 4780) instead — there are TWO entry points and only the latter is wired.

The "Phase-1 skeleton — bodies stubbed" comment at header lines 6–11 is a **stale TLDR from the original scaffold commit** that was never updated. Codex was right to flag it. The actual code has been through Phases 2, 3, 4, 4.1, 4.2, 4.3, 4.4 (a-d), 4.5 (Steps 1-4), and §5.5 (sub-steps 1-13 of Step 4 hardening).

---

## Phase status table

| Phase | Header marker | Status | Citation |
|---|---|---|---|
| 1 | "Phase-1 skeleton" intro | **STALE** — describes original 2024-Q3 scaffold; never refreshed | h:6–11 |
| 2 | `init_from_gguf` weight upload | **COMPLETE** (Q2.1 Q4-resident path) | h:339, cpp:721–1235 (514 LOC) |
| 2.1 | Q4-resident packed-INT4 + F16 scale tile | **COMPLETE** | cpp file header :2 |
| 3 | `forward_block_` real compute | **COMPLETE** (~15 aclnn ops/block) | h:711, cpp:3037–3837 (800 LOC) |
| 4 | denoise loop wiring (canonical `denoise()`) | **STUB** — returns false with log | h:362, cpp:1264–1285 |
| 4.1 | On-device RoPE | **COMPLETE** (host fallback preserved via `QIE_ROPE_HOST=1`) | h:834, cpp:2620–3036 |
| 4.2 | `forward_all_blocks_test` 60-block stack | **COMPLETE** | h:410, cpp:4182–4267 |
| 4.3 | Euler scheduler + `denoise_loop_test` | **COMPLETE** (cos_sim=1.0 vs CPU ref on synthetic) | h:421/450, cpp:3863–4155 |
| 4.4 | Real-Q4-GGUF + F32 residual stream | **COMPLETE** (Q2.4.4d landed F32 residual fix) | h:614, cpp:2098–2216 |
| 4.5 | `denoise_full` production entry + `init_from_dump` | **COMPLETE** | h:460/510, cpp:4655–5645 (865 LOC) |
| 4.5.4c-d | BF16 plumbing for ff_down + attn-out residual contributors | **COMPLETE** under `QIE_ALL_BF16=1` | h:665, cpp:3037–3837 |
| §5.5 | NaN+visual hardening (Steps 4c→4l, 13 sub-iters) | **PARTIAL** — NaN gate GREEN, visual gate still RED | docs/qie_q2_phase4_smoke.md §5.5.5–§5.5.13 |

---

## Method-by-method classification (.cpp)

| Method | Body lines | LOC | Classification | Notes |
|---|---|---|---|---|
| `~ImageDiffusionEngine` | 631–719 | 89 | **COMPLETE** | Full device-buffer teardown |
| `init_from_gguf` | 721–1235 | 514 | **COMPLETE** | GGUF parse + Q4-resident upload + RoPE table build + scratch allocation |
| `forward` | 1241–1259 | 18 | **COMPLETE** | Loops `forward_block_` over `cfg_.num_layers` |
| `denoise` (canonical) | 1264–1285 | 22 | **STUB** | Logs "scaffold Phase 2", returns false. Production calls `denoise_full` instead. |
| `alloc_dev_` / `ensure_workspace_` | 1290–1335 | 46 | **COMPLETE** | |
| `build_rope_tables_` | 1336–1344 | 8 | **NOOP** (subsumed by init_from_gguf) | Documented; not a real bug |
| `build_time_emb_` | 1345–1374 | 30 | **COMPLETE** | Sinusoidal 256→f16 |
| `dispatch_matmul_` | 1460–1907 | 448 | **COMPLETE** | WQBMMv3 + aclnnMm fallback + BF16 output path |
| `modulate_` | 1909–1985 | 77 | **COMPLETE** | |
| `gated_residual_add_` | 1987–2043 | 57 | **COMPLETE** (F16 path) |
| `gated_residual_add_f32_` | 2098–2216 | 119 | **COMPLETE** (Phase 4.4c F32 accumulator) |
| `gated_residual_add_f32_bf16src_` | 2218–2342 | 125 | **COMPLETE** (Q2.4.5.4c BF16-src) |
| `cast_f32_to_f16_` | 2045–2096 | 52 | **COMPLETE** |
| `layer_norm_` | 2344–2402 | 59 | **COMPLETE** (affine-off) |
| `layer_norm_f32_to_f16_` | 2404–2482 | 79 | **COMPLETE** (Phase 4.4c) |
| `rms_norm_row_f32_to_f16_` | 2484–2551 | 68 | **COMPLETE** (Phase 4.5 Step 4) |
| `rms_norm_head_` | 2553–2618 | 66 | **COMPLETE** |
| `apply_rope_` (dispatcher) | 2620–2644 | 25 | **COMPLETE** |
| `apply_rope_host_` | 2646–2758 | 113 | **COMPLETE** (Phase-3 baseline, retained as fallback) |
| `apply_rope_on_device_` | 2760–2856 | 97 | **COMPLETE** (Phase 4.1) |
| `apply_rope_manual_` | 2858–3035 | 178 | **COMPLETE** (manual 4-Mul + 2-Add fallback) |
| `forward_block_` | 3037–3837 | **800** | **COMPLETE** | 15+ aclnn ops/block, F32 residual, BF16 leak sites, Q/K-RMSNorm + RoPE + FIA-or-MM-softmax-MM, FFN |
| `scheduler_step_` | 3839–3855 | 17 | **NOOP** (canonical body deferred — production uses `scheduler_step_test`) |
| `scheduler_step_test` | 3863–3913 | 51 | **COMPLETE** |
| `denoise_loop_test` | 3943–4155 | 213 | **COMPLETE** (cos_sim=1.0 GREEN on synthetic) |
| `forward_block_test` | 4161–4180 | 20 | **COMPLETE** |
| `forward_all_blocks_test` | 4182–4267 | 86 | **COMPLETE** |
| `mutable_layer_weights` | 4269–4272 | 4 | **COMPLETE** |
| `init_for_smoke` | 4274–4453 | 180 | **COMPLETE** |
| `init_from_dump` | 4655–4778 | 124 | **COMPLETE** |
| `denoise_full` | 4780–5645 | **865** | **COMPLETE** | Production 20-step loop: time_embed→img_in→60 blocks→norm_out→proj_out→host unpatchify→flow-Euler |

**Stub count:** 2 of 31 (canonical `denoise` + canonical `scheduler_step_`). Both are documented as "use the test/full variants instead". Not blockers.

---

## Smoke test evidence (read-only, from existing logs on ac03)

**Did NOT run a new smoke** — ac03 is currently idle (only `tail -F` processes), but existing artefacts already prove the engine runs end-to-end:

- Binary: `/home/ma-user/work/OminiX-Ascend/tools/probes/qie_q45_step4_full_denoise/test_qie_q45_step4_full_denoise` (1.8 MB, mtime 2026-04-26)
- Object: `build-w1/tools/qwen_image_edit/CMakeFiles/qwen_image_edit_native.dir/native/image_diffusion_engine.cpp.o` exists
- Run log `/tmp/qie_q45_step4f_precise.log` tail (most recent §5.5.6 run):
  ```
  [smoke45s4] dispatching denoise_full (real Q4_0 weights, real host conditioning, 20-step flow Euler, cfg=1.00)...
  [qie_native] denoise_full: W_lat=32 H_lat=32 C_lat=16 B=1 img_tokens=256+256=512 txt_seq=214 joint_dim=3584 n_steps=20 cfg=1.00
  ...
  [smoke45s4] denoise_full OK (24770.59 ms)
  per-step min=1169.80 median=1173.55 max=1305.24
    out_latent: mean=-2.4559 std=4.8611 min/max=-13.3203/7.6250 NaN=0 inf=0
  VERDICT: GREEN  (gate: NaN=0, inf=0, std>0.0010, |min|<20, |max|<20)
  ```
- Output PNG: `/tmp/qie_q45_step4d_allbf16_cat.png` (110 KB, 256×256 RGB) — coherent tile pattern, **not** a recognizable cat. §5.5.6 documents this is the open visual gate.
- Block-0 substep oracle (§5.5.13): `cos=1.000000` for all six QKV projections vs analytical Q5_K oracle — engine math validated bit-for-bit at the projection layer.

---

## Why Branch B (and not A or C)

**Not Branch A:** the canonical 256² eye-check still emits a tile-pattern PNG, not a cat. `cos≈0.48` at substep 11 (`attn_out`) and substep 24 (`resid2`) vs CPU reference. §5.5.6 calls this "host unpatchify or pe-table layout artifact" but the bisect (§5.5.7–§5.5.13) has been chasing it for ~13 sub-iterations without yet closing. Switching production CLI to native NOW would ship recognizable-image regressions.

**Not Branch C:** the engine is far too real to be dead code — 5,647 LOC of compute, 514 LOC weight upload that completes in 101 s and is 20% of the unpushed-ahead commits, GREEN NaN gate on real GGUF, and active development under §5.5.x. The §5.5.13 oracle proof (cos=1.0 vs analytical Q5_K) is decisive — the engine produces correct numerics through QKV projection at least.

**Why Branch B fits:** the engine works, is the obvious endgame, but is not visually correct today. The CLI / ggml-cann path (`§5.5.67` first-NaN trace) is still the only path that has produced a recognizable cat-edit at 1024². Both must coexist until the native engine clears its visual gate.

---

## Recommendation

**Single recommended next step:** finish the §5.5 visual-gate bisect in the native engine, NOT in the CLI ggml-cann path. The native engine is closer to production than ggml-cann (NaN gate GREEN, only attention/unpatchify drift remains), and the §5.5.x agents are already inside it. Specifically:

1. **Park §5.5.67** (CLI ggml-cann first-NaN trace) — its work is duplicative once native lands. Keep ggml-cann working as today's fallback but stop investing.
2. **Resume from §5.5.13's "Open" item:** "end-to-end denoise → VAE → PNG to confirm the win condition." The §5.5.13 substep recovery (08 cos 0→0.80, projection oracle cos=1.0) is a strong signal the upstream bug is now in attention or unpatchify, not projection. Run the next bisect there.
3. **Once native cat-PNG is recognizable**, retire `denoise()` as a stub-with-redirect (or wire it to call `denoise_full`), retire CLI ggml-cann path from production-config defaults, and the §5.5.x saga closes.

**Integration cost when native is GREEN visually:** the CLI side (`tools/ominix_diffusion/cli/main.cpp`) currently drives the ggml-cann graph; the native engine has its own driver `tools/qwen_image_edit/native/main_native.cpp` (152 LOC). Merging them behind a `--native` flag is ~50 LOC of arg parsing + `ImageDiffusionEngine::denoise_full` call site. Not a multi-day task once the visual gate clears.

---

## Open contradictions resolved

- Header lines 6–11 ("Phase-1 skeleton — bodies stubbed") describe the original scaffold and were not updated. The TRUE current state is at header lines 460–520 (Phase 4.5 Step 4 production `denoise_full`), 614–627 (Phase 4.4c F32 residual), and 665–680 (Q2.4.5.4c BF16 plumbing).
- Two `denoise` symbols exist: canonical `bool denoise(...)` is a stub (returns false) and **production code uses `bool denoise_full(...)` instead.** Anyone reading only the canonical entry point would conclude Branch C; anyone reading `denoise_full` and the §5.5 saga would conclude Branch A. The truth is between: Branch B.
