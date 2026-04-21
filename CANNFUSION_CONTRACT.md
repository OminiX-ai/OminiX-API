# CannFusion Pivot — Qwen3-TTS FFN / Matmul Fusion via Codegen

## 1. Status & mandate

**Status (2026-04-21 pm, updated)**: **PARKED — WAITING UPSTREAM**.
F0 toolchain probe PASS on CANN 8.3.RC1 / Ascend 910B4 (build, generate,
compile, device-dispatch all clean via bisheng/ccec/lld — cannfusion
0.2.0). However F0 surfaced upstream `CANN_ESCALATION.md §1.12` (dated
2026-04-17, open after 4 fix waves through Wave 12e''', PM note
`1dcb3c3`): **fused-epilogue cube-only path hits
`ACL_ERROR_RT_AICORE_EXCEPTION 507015` on `aclrtSynchronizeStream`**.
The fault is a CANN *runtime* bug — "runtime decodes vec-arch
instructions as cube-arch instructions" — not patchable in CannFusion's
codegen. Resolution depends on Huawei CANN engineering cadence.

The entire pivot value-add (silu + mul + residual-add epilogue fusion
for the CP FFN sublayer) is blocked on §1.12. Plain matmul (empty
epilogue) is convergent per §1.9 but has near-zero fps upside vs the
existing `aclnnWeightQuantBatchMatmulV3` path — not worth dispatching
F1 just for that.

PM has pivoted again to the `aclGraph` track (see
`aclgraph_feasibility.md` once Agent G0 reports). This contract stays
in-tree as a **watchpoint**: resume F1/F2 if upstream merges a §1.12 fix
in a future CannFusion release.

**Origin**: PM decision to pivot from Path C hand-written AscendC after
W4.1.3 landed on fork (commit `3ac9aab5`) but showed 6.5× slowdown
preview under the naive `blockDim=1` vector kernel. Tiling to 20-core
parallelism is days of work with uncertain outcome; CannFusion's codegen
path is already authored, Apache-2.0, and targets exactly the
Mm-+-epilogue fusion we need for the FFN sublayer — **but the fusion
half is currently broken upstream**.

**Ceiling claim to beat**: current clean-quality canonical xvec mayun zh
delivers **30.5 fps** (W8 + TQE=2 + cp_groups=15 + sampling + W1 NPU
lm_head + W3b cp-fusion). The goal remains **≥ 32 fps** on the same
canonical benchmark with identical user-ear verdict ("no rumble, clean
voice"). Any regression in voice quality is a failing gate regardless of
fps.

**PM role**: supervise; each workstream author/agent works solo on ac01
or designated container, reports at gates, PM arbitrates. PM does not
write kernel code.

## 2. Background — why pivot

Path C W4.1.3 (hand-written fused attn sublayer AscendC kernel)
compiled, wired, ran without crash, but preview walltime shows 215
ms/frame ASCENDC-on vs 33 ms/frame baseline. Root cause: kernel uses
`blockDim=1` — single vector core out of ~20 available — no tiling, no
double-buffer, no multi-core dispatch. Optimising that would be days of
work re-authoring the kernel.

CannFusion (https://gitcode.com/Rust4CANN/CannFusion) is a Rust-hosted
build-time codegen that emits full AscendC operator directories
(`kernel.h`, `kernel_entry.cpp`, `tiling.cpp`, `binary.json`,
`CMakeLists.txt`, plus ACLNN two-phase host API) for a narrow but
load-bearing class of kernels: **GEMM + fused epilogue**. Generated
kernels use the Cube Unit systolic array directly, with autotune-selected
tile shapes. Runtime dispatch surface is identical to a hand-written
kernel (same `aclrtlaunch_<kname>` host wrapper, same `ascendc_library`
cmake kit we validated at W4.0).

Prior assessment: `/Users/yuechen/home/OminiX-Ascend/docs/cannfusion_assessment.md`
(Agent CF-research, 2026-04-21). TL;DR capability fit:
- **YES** for FFN GEMM + epilogue (gate-Mm + SiLU + residual-add, etc.)
- **NO** for attn (RmsNorm-prologue + RoPE-mid-kernel + FIAS not in scope)
- **UNKNOWN** CANN 8.3.RC1 compat (project likely developed on 8.5)
- **UNKNOWN** A16W8 (F16 × INT8 → F16) — validator matrix lists INT8×INT8

The W4.1.3 attn kernel stays landed (env-gated off by default via
`TALKER_CP_ASCENDC`); no revert needed. Future return to Path C attn
tiling is not pre-funded but left possible.

## 3. Scope

**In scope**:
- F0 probe: does CannFusion generate and compile on CANN 8.3.RC1 / ac01
  with bisheng available? HARD GATE — if no, adopt fallback strategy.
- F1 probe: does CannFusion validator accept our `A16W8` dtype
  (F16 activation × INT8 weight → F16 output, per-channel dequant)?
- F2: fuse CP-layer FFN gate / up path into one Mm+epilogue kernel,
  replacing the stock `aclnnWeightQuantBatchMatmulV3 + aclnnSiLU + mul`
  chain.
- F3: fuse CP-layer FFN down path (Mm + residual-add) into a second
  generated kernel.
- F4: perf gate ≥ 32 fps on canonical xvec mayun zh.
- F5: user-ear gate vs pre-CannFusion baseline wav.

**Out of scope**:
- Attn sublayer fusion (stays in Path C land as `TALKER_CP_ASCENDC`).
- Replacing FIAS, RmsNorm standalone ops, RoPE — not in CannFusion.
- Talker (non-CP) transformer work — separate contract if needed.
- Autotune sweeps beyond first-pass default tile.

## 4. Host plan

- **Primary**: `ac01` (port 31984). CANN 8.3.RC1. Existing fork at
  `~/work/OminiX-Ascend-w1/`. Already has bisheng + ascendc_library
  cmake kit validated at W4.0.
- **Secondary**: `ac02` (31210) or `ac03` (30412) available for
  parallel generated-smoke runs without blocking ac01.
- **Build-time toolchain**: Rust 1.85 stable + `cargo install cannfusion`.
  Not on ac01 kernel host. Ships offline as a vendored source drop into
  `tools/qwen_tts/ascendc/cannfusion_gen/` — zero runtime Rust footprint.
- **Fork**: `https://github.com/ymote/OminiX-Ascend` `main`. ac01 has no
  git push creds → same patch mechanism as Path C: `git format-patch` on
  ac01 + scp to Mac + PM pushes.

## 5. Workstreams with gates

### F0 — CANN 8.3.RC1 toolchain probe (1 hour, HARD GATE)

- [x] F0.1 Clone + cargo build on Mac. **PASS** — cannfusion 0.2.0,
      Rust 1.85 MSRV, CLI at `target/release/cannfusion`.
- [x] F0.2 Generate from `tests/fixtures/valid_basic.toml`. **PASS** —
      12 files produced (kernel.h, kernel_entry.cpp, kernel_meta.cpp,
      tiling_data.h, tiling.cpp, def.cpp, infershape.cpp, binary.json,
      api.h, api.cpp, CMakeLists.txt, vec_kernel.h).
- [x] F0.3 Compile on ac01 / CANN 8.3.RC1 / Ascend 910B4. **PASS** —
      bisheng clang 15.0.5, CANN 8.3.0.1.200, full cmake build ≤ 10 s.
      Two-pass bisheng (cube + vec) + `ld.lld -m aicorelinux` produces
      fatbin `kernel_entry.o` + `libcannfusion_gemm_host.so` with
      exports `aclnnCannfusionGemm` / `aclnnCannfusionGemmGetWorkspaceSize`.
- [x] F0.4 Device-dispatch smoke. **PASS-WITH-CAVEAT** —
      `LaunchAscendKernel rc=0`, stream sync returns `507015`
      (`ACL_ERROR_RT_AICORE_EXCEPTION`). The `507015` is EXPECTED per
      `api.cpp` L196-221: the aclnn public API is a v0.1.1 shim that
      requires caller-populated tiling blob (wave-13 TODO upstream).
      Project's own `scripts/device-smoke/runner.cpp` populates
      `CfTiling` by hand and reports `0/65536 mismatches, max_abs=0` on
      this exact ac01/CANN/SoC triple (see `CANN_ESCALATION.md §1.9`).

**Gate verdict**: **PASS** (toolchain compatible with 8.3.RC1). BUT
F0.4 surfaced the §1.12 bug (see §1 STATUS): fused-epilogue cube-only
cmd-stream throws the same 507015 even when tiling IS populated, and
this is an unresolved CANN runtime bug upstream has escalated to
CANN engineering. F1/F2 NOT dispatched — fused epilogue is the entire
pivot value prop, and it is the surface that's blocked.

### F1 — A16W8 dtype probe (1-2 hours, HARD GATE)

- [ ] F1.1 Author minimal TOML config for `F16 × INT8 → F16`
      matmul with per-channel F16 scale. If validator rejects via
      `ValidationError` — confirm error content, check whether
      CannFusion's source has a `dtype_compat` matrix we can extend.
- [ ] F1.2 If validator accepts: generate + compile + run device-smoke
      comparing output vs our existing `aclnnWeightQuantBatchMatmulV3`
      on a 2048×896 × 896×4864 matmul (matches CP FFN gate shape).
- [ ] F1.3 If validator rejects: evaluate fallback — either
      (a) upstream patch to CannFusion (slower, needs repo contact), or
      (b) pass our W8 weights as INT8×INT8 with act-quant baked in
      (would need A8W8 path not currently available in our engine), or
      (c) abandon CannFusion — PM decision.

**Gate**: numerical parity ≤ 1e-2 vs reference matmul on shapes we
actually use in CP FFN.

### F2 — FFN gate/up + SiLU-mul kernel (~3 days)

- [ ] F2.1 TOML config for the CP FFN "gate ⊙ silu(gate)" shape.
      Shape: X=[1, cp_hidden=896], W_gate=[cp_hidden, inter=4864],
      W_up=[cp_hidden, inter], epilogue = SiLU(gate_out) ⊙ up_out. One
      option: two separate generated kernels (gate-Mm + SiLU, up-Mm +
      residual-prep); second kernel does the mul as its prologue.
      Alternative: one Mm, stream the second via elementwise epilogue
      of first kernel. Author picks in F2.1.
- [ ] F2.2 Generate, compile, device-smoke numerical parity vs current
      chain ≤ 1e-2.
- [ ] F2.3 Wire into `cp_cann_engine.cpp` behind `TALKER_CP_CANNFUSION=1`.
      Unset = stock path bit-identical.
- [ ] F2.4 Live token drift ≤ 1 / frame / group, first 16 frames
      canonical xvec mayun zh (same gate as W1.4 and W4.1.4).
- [ ] F2.5 Perf preview: fwd time before/after. Report.

### F3 — FFN down + residual-add kernel (~2 days, parallel to F2 tail)

Same structure as F2 for the "down-Mm + residual-add" half.

### F4 — Perf HARD GATE (no new code; bench only)

- [ ] F4.1 Canonical xvec mayun zh with all env vars that matter:
      `TASK_QUEUE_ENABLE=2 TALKER_W8_QUANT=1 TALKER_LM_HEAD_NPU=1
      TALKER_CP_FUSION=1 TALKER_CP_CANNFUSION=1`. Seed 42, cp_greedy,
      full-length text. Measure fps over 3 runs, take median.
- [ ] F4.2 Gate: fps **≥ 32** on the canonical benchmark. If missed,
      STOP, report, PM decides: invest more (F-optional kernels) / ship
      at-current-fps / fall back to 30.5 baseline and close contract.

### F5 — User-ear gate

- [ ] F5.1 Deliver two wavs to `/Users/yuechen/Downloads/tts_ab/`:
      `cf_final.wav` (CannFusion ON) and `cf_pre.wav` (OFF, =
      current 30.5 fps baseline). PM listens.
- [ ] F5.2 PM sign: voice quality must match pre-baseline. Any
      regression (rumble, distortion, new artefacts) = REJECT; roll the
      env gate to unset-default and ship without CannFusion.

### F-optional — Non-CP matmuls

Only if F4 passes and leaves headroom below 33 fps (per W4.2/W4.3
contract pattern). Targets Talker QKV / O / FFN matmuls. Not pre-funded.

## 6. Acceptance criteria (summary)

- [ ] F0 + F1 gates cleared; CannFusion is known-compatible with 8.3.RC1
      and our A16W8 dtype.
- [ ] F2 + F3 kernels land on fork env-gated by `TALKER_CP_CANNFUSION=1`.
- [ ] F4 perf gate ≥ 32 fps on canonical benchmark.
- [ ] F5 user-ear gate: no quality regression.
- [ ] Contract updated with a single verified `Verified-by` stamp per
      workstream including commit SHA.

## 7. Risks

1. **CANN 8.3.RC1 incompat** — CannFusion may use APIs not present in
   our CANN. Mitigation: F0 probe first. Fallback: upstream patch (days)
   or abandon.
2. **A16W8 dtype missing** — validator may reject. Mitigation: F1 probe.
   Fallback per F1.3.
3. **No measured win** — fps parity at best, regression at worst if the
   fused kernel's tile shape is bad. Mitigation: F4 is HARD GATE; we
   roll back on miss.
4. **Autotune on device** — if default tile is suboptimal we could
   autotune, but each autotune run is ~minutes. Keep first-pass default
   unless F4 is close to passing; don't open a tuning rabbit hole until
   then.
5. **Silent quality regression** — even bit-identical matmul output can
   compound through FFN into audible artefacts. Mitigation: F5 ear gate
   is mandatory, not advisory.
6. **CannFusion project maturity** — v0.2.0, unknown contributor count.
   We're adopting a young dependency. Mitigation: vendor the source
   in-tree at F0.1; if upstream stalls we fork.

## 8. Host rules (carried from Path C)

- All kernel compile work on ac01. Mac has no CANN headers; LSP errors
  under `cp_cann_engine.{h,cpp}` on Mac are expected and noise.
- No force-push to fork `main` without PM authorisation. All merges via
  PM's Mac.
- Each gate stops the agent for PM review. No silent continuation past
  a failing gate. Perf gates (F4) are HARD KILL.
- No Claude coauthor on commits.
- Token-drift and user-ear gates outrank fps numerics. A wrong 40 fps
  ships nothing; a clean 30 fps ships.
