# Path C: AscendC Custom-Kernel Contract

Follow-on to `CP_FPS_OPTIMIZATION_CONTRACT.md`. W3b delivered only
+0.66 fps because `TASK_QUEUE_ENABLE=2` already amortizes per-dispatch
launch overhead — stock `aclnn` fusion can't close the remaining gap.
Path C authors custom AscendC kernels that **keep intermediate tensors
in on-chip SRAM** between multiple ops, eliminating DRAM round-trips
that stock `aclnn` always incurs.

**PM role**: supervise, gate, arbitrate. **PM does not write code.**
Every milestone below is an agent deliverable. PM signs off by
verifying the gate criteria (numbers, commit hash, ear-check).

## 1. Goal (single sentence)

Lift clean-quality xvec mayun zh fps from **30.5 fps** (current
committed, `clean-30.5fps-ear-verified` tag) to **≥ 33 fps** on ac01
via one or two custom AscendC kernels fusing CP-transformer
sublayers, **without regressing audio quality** (user-ear identical,
token drift ≤ 1 / frame / group).

## 2. Non-goals

- Not rewriting the full CP transformer in AscendC.
- Not touching Talker forward (separate track).
- Not the ICL Talker-KV gap (separate track).
- Not revisiting Path A/B (W3b closed that).
- Not MRoPE / xvec / customvoice logic changes.
- Not flipping any default — new code stays behind `TALKER_CP_ASCENDC=1`.

## 3. Current state (2026-04-21)

Committed: W1 (NPU lm_head) + W3b (`aclnnAddRmsNorm` fusion). Tag
`clean-30.5fps-ear-verified` on fork `ymote/OminiX-Ascend`.

**Per-frame CP cost** (mayun xvec zh, W1+W3b on, `TALKER_CP_PROF=1`):
- lm_head (NPU, via W1): ~2 ms
- CP `forward_one_token` (NPU, ×15): 18.3 ms each
- Total CP phase: ~20.3 ms → 30.5 fps wall

**Remaining dispatches per forward_one_token** (post-W3b): ~80 compute
ops. Attention sublayer = 12 ops; FFN sublayer = 7 ops.

## 4. Architecture target

**Two fusion targets**, each a candidate for one AscendC kernel:

**Kernel A — fused attention sublayer**. Collapses:
```
RmsNorm → [Q,K,V]-Mm(W8) → RoPE → FIAS → O-Mm → +residual
```
12 aclnn dispatches → 1 custom kernel. Q/K/V/attn intermediates stay in
SRAM (no DRAM). Expected savings: **~2 ms / forward_one_token** on the
W3b-post baseline.

**Kernel B — fused FFN sublayer**. Collapses:
```
RmsNorm → [gate,up]-Mm(W8) → SiLU → ⊙ → down-Mm → +residual
```
7 aclnn dispatches → 1 custom kernel. Expected savings: **~1.2 ms /
forward_one_token**.

Combined (both kernels): ~3.2 ms / forward_one_token × 15 groups =
48 ms / frame → theoretical ceiling ~40 fps. Realistic target with
kernel overhead: **33-35 fps**.

Env gate: `TALKER_CP_ASCENDC=1`. Both kernels opt-in; unset restores
bit-identical W1+W3b path.

## 5. Milestones

### W4.0 — Toolchain go/no-go (half day)

Single-track probe. **Agent C-probe**, ac01. If any sub-step fails,
STOP. Path C becomes a toolchain escalation, not an engineering track.

- [ ] 4.0.1 Verify `ccec` AscendC compiler on ac01
      (`$ASCEND_TOOLKIT_HOME/compiler/ccec_compiler/bin/ccec --version`
      works; path is already in PATH per `/etc/profile` — confirm).
- [ ] 4.0.2 Verify `aclnn` custom-op registration API: look for
      `aclopRegisterCustom`, `aclopCreateHandle`, or equivalent in
      `$ASCEND_TOOLKIT_HOME/aarch64-linux/include/acl/`. List what's
      available; record which symbols are dlsym-accessible.
- [ ] 4.0.3 Write **hello_kernel.ccec** (~20 lines, identity copy
      input→output). Compile with `ccec`, package as `.so`, `dlopen`
      from a 40-line C++ test binary, dispatch on ac01, verify output
      matches input.
- [ ] 4.0.4 Record: compile command, package shape, dispatch boilerplate,
      timing vs plain `aclrtMemcpy`. This is the template the W4.1 /
      W4.2 kernels will extend.

**W4.0 acceptance**: hello_kernel round-trip works on ac01; agent
reports the exact build + dispatch commands as a reusable snippet.
PM gates here — if infra isn't in place, no further Path C spend.

**Open issues for W4.0**:
1. Does CANN 8.5 require kernels to be pre-registered via tiling
   functions, or can AscendC kernels be dispatched ad-hoc? If former,
   tiling-fn authoring adds ~1 day per kernel — add to W4.1/W4.2 scope.
2. Is there an `aclrtLaunchKernel` alternative that accepts AscendC
   binaries directly?
3. Do our `TALKER_W8_QUANT=1` INT8 weights need re-layout for AscendC
   kernel access (vs the aclnn layout the engine uses today)?

### W4.1 — Fused attention sublayer kernel (~2 weeks)

Single-track, ac01. **Agent C-attn**.

- [ ] 4.1.1 **Kernel skeleton**: author `fused_attn_sublayer.ccec`
      implementing: one block of (RmsNorm → QKV-matmul → RoPE →
      attention → O-matmul → residual-add). F16 arithmetic. Weight
      layout matches `cp_cann_engine.cpp`'s existing Q/K/V/O uploads;
      **do not re-pack weights**, the engine's existing tensors are
      the input.
- [ ] 4.1.2 **Offline numerical validation**: extract one layer's
      input hidden + Q/K/V/O weights into test fixture. Dispatch the
      kernel on that fixture and compare output to the stock aclnn
      chain's output. Max-abs-diff ≤ 1e-2 (F16 noise). This step runs
      against canned input, not the live generate loop.
- [ ] 4.1.3 **Wire into `CpCannEngine`**: replace the attn-sublayer
      aclnn chain with `forward_lm_head`-style dispatch of the fused
      kernel, gated on `TALKER_CP_ASCENDC=1`. Unset = old path
      bit-identical. Skip dispatching both paths in parallel — gate is
      compile-time-like (runtime env, but path-exclusive).
- [ ] 4.1.4 **Correctness gate** (live): canonical mayun xvec zh under
      `--cp_greedy --seed 42`. Token drift ≤ 1 / frame / group for
      first 16 frames (same gate as W1.4). If > 1, debug — do not
      ship.
- [ ] 4.1.5 **Perf gate**: `TALKER_CP_PROF=1` shows fwd drops from
      18.3 ms → **≤ 16 ms / forward_one_token**. fps on canonical
      lifts from 30.5 → **≥ 32 fps**. If gate misses, STOP and report
      — PM decides to abandon Path C or adjust scope.
- [ ] 4.1.6 **User-ear gate**: deliver
      `/Users/yuechen/Downloads/tts_ab/w4a_final.wav` (attn-fusion ON)
      + `w4a_pre.wav` (off). PM ear-checks.

**Open issues for W4.1**:
1. What's the attn tile / pipeline strategy? AscendC samples show
   L1-resident Q-tiles × DRAM-streamed K/V/O; adopt that unless
   measurements say otherwise. Agent C-attn proposes in 4.1.1.
2. How does the fused kernel handle KV-cache append? Engine tracks
   `current_pos`; the kernel either takes (`pos`, existing-KV-pointer)
   and writes the new K/V slot itself, or engine appends externally
   and kernel reads the full KV. Pick one; document trade in 4.1.1.

### W4.2 — Fused FFN sublayer kernel (~1 week, gated on W4.1 passing)

Only proceeds if W4.1.5 cleared ≥ 32 fps. Same structure.

- [ ] 4.2.1 Kernel skeleton `fused_ffn_sublayer.ccec`
- [ ] 4.2.2 Offline numerical validation (≤ 1e-2)
- [ ] 4.2.3 Wire into CpCannEngine behind same `TALKER_CP_ASCENDC=1`
      gate
- [ ] 4.2.4 Correctness gate (live): ≤ 1 token drift / frame / group
- [ ] 4.2.5 Perf gate: cumulative fps ≥ **33 fps** xvec mayun canonical
- [ ] 4.2.6 User-ear gate: `w4b_final.wav` vs `w4b_pre.wav`

### W4.3 — (Optional) Cross-sublayer super-kernel (only if W4.1+W4.2 leave headroom)

If W4.2 lands cleanly and fps is at e.g. 33.5 (below 35), consider
one more spike fusing attn + FFN into a single super-kernel per
layer. PM decision point; not pre-funded.

## 6. Acceptance criteria (summary)

- [ ] W4.0 toolchain probe green
- [ ] W4.1 lands on fork with `TALKER_CP_ASCENDC=1` gate; xvec mayun zh
      ≥ 32 fps; token drift ≤ 1 / frame / group; user-ear clean
- [ ] W4.2 lands (if W4.1 passed); cumulative ≥ 33 fps; drift + ear
      clean
- [ ] Default off; unset gate = bit-identical to `clean-30.5fps-ear-verified`

## 7. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| `ccec` missing or doesn't package as a reusable `.so` | High | W4.0.3 hello-kernel probe catches this in half a day. |
| Custom-op registration API not dlsym-accessible from existing `cp_cann_symbols.cpp` path | Medium | W4.0.2 records what IS available; agent picks the working API. |
| Weight re-layout needed (AscendC wants different INT8 layout than aclnn stores) | Medium | If yes, one-time re-pack at engine init (not per-frame). W4.1.1 documents layout; 4.1.2 validates against that layout. |
| Fused kernel slower than aclnn chain (bad tile strategy) | Medium | W4.1.5 perf gate is the kill criterion. PM aborts rather than iterate indefinitely. |
| F16 precision drift accumulates across kernel boundaries | Medium | 4.1.2 offline diff + 4.1.4 live drift + 4.1.6 ear gate all enforce; intermediate re-cast to F32 if needed. |
| AscendC kernel OOB / crashes on live NPU | Low | Develop behind env gate; never flip default; validate on short utts before long utts. |

## 8. Parallelism playbook

**Single-track, ac01 only.** The CP engine's source files
(`cp_cann_engine.cpp`, `cp_cann_symbols.cpp`) will be edited by both
W4.1 and W4.2 — running those in parallel would collide. Sequential
is the only safe path.

**Exception**: W4.2.1 (FFN kernel skeleton authoring) CAN overlap
with W4.1.6 (user-ear check on attn kernel), because those don't
touch the same files. Agent C-ffn can start sketching while PM is
running user-ear on attn. PM authorizes this overlap only once
W4.1.5 perf gate has passed.

**Host assignments**:
- ac01: all W4 milestones (kernel development + ac01 has the build
  tree + NPU for dispatch)
- ac02 / ac03: unused. Available for exploratory kernel
  micro-benchmarks if Agent C-attn wants a sandbox that doesn't
  touch the main build.

**Agent dispatch cadence (PM)**:
- One agent per milestone. 8 milestones total (0.1-0.4, 1.1-1.6, 2.1-2.6).
- Longer milestones (4.1.1 kernel authoring) may span multiple agent
  dispatches if the agent hits its session context limit; PM resumes
  or spawns a fresh agent with the prior agent's hand-off notes.

## 9. Sign-off

- [ ] User signs this contract (PM requests before any code).
- [ ] On sign-off, PM dispatches Agent C-probe for W4.0.
- [ ] PM gates each milestone: verifies numbers + ear-check artifacts
      before authorizing the next milestone.
- [ ] PM pulls the plug at W4.1.5 if the perf gate doesn't clear.

---

**PM role reminder**: supervise, not implement. Every milestone is an
agent deliverable with a concrete quantitative gate. PM reads
reports, checks numbers, verifies ear, signs off, moves to next.

**Scope honest-sizing**: 3-4 weeks wall-time. Agent budget: ~8 agent
dispatches plus PM gate meetings between each.

**Current status**: draft, pending user sign-off. No code spawned.
