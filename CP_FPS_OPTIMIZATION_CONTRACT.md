# CP FPS Optimization Contract

Workstream to lift the clean-quality TTS ceiling from **22 fps** (current
committed state at `8038c335`) toward **30 fps** without sacrificing the
ear-validated audio quality settings (W8 + TQE=2 + cp_groups=15, sampling
on). Two parallel tracks; agent swarm executes, PM (this session) signs
off per milestone.

## 1. Goal

Lift clean-quality fps ceiling from 22 → **≥ 28 fps** on the canonical
long-utt ICL benchmark (mayun_ref Chinese, seed=42, W8+TQE=2,
cp_groups=15, sampling on), with **byte-identical or ASR-identical
audio** to the current 22 fps baseline. Reach **30 fps** if both tracks
land.

## 2. Non-goals

- Not reducing cp_groups (user already rejected — rumble).
- Not switching to greedy decoding (user already rejected — distortion).
- Not W8→F16 revert (user confirmed W8 is fine when cp_groups=15).
- Not touching the decoder / tokenizer paths (out of scope; not the bottleneck).
- Not expanding the C ABI (bridge contract's surface stays frozen).

## 3. Current state (as of 2026-04-20)

**Per-frame CP cost on canonical xvec mayun (`TALKER_CP_PROF=1`):**

| phase | time | location | addressable? |
|---|---|---|---|
| `lm_head` matvec (15×) | 14-17 ms | CPU NEON+OMP (at peak) | ✅ W1 |
| CP `forward_one_token` (15×) | 17 ms | NPU (aclnn) | ✅ W2 |
| `sample_token` (15×) | 2 ms | CPU | ❌ — too small to matter |
| `read_embedding_row` | 0.2 ms | CPU memcpy | ❌ — negligible |
| **Total CP path per frame** | **~33 ms** | | |
| Talker forward (1×) per frame | ~10 ms | NPU | ❌ — out of scope |
| **Wall per frame** | **~45 ms → 22 fps** | | |

**Two addressable costs**: lm_head (15 ms) and CP forward (17 ms). Combined
potential saving: up to 24 ms/frame → 47 fps ceiling if both dropped to
zero (impossible but bounds the ROI).

## 4. Architecture target

**W1 — NPU lm_head port.** Upload the 15 lm_head weights to NPU at load.
Add `CpCannEngine::compute_logits(group_idx, cp_out_device, logits_device)`
that dispatches one `aclnnMm` per call. Fetch only the final `vocab_size=2048`
float buffer back to CPU for sampling. Saves ~12 ms/frame (lm_head drops
from 15 ms → ~3 ms: 1 ms aclnn dispatch × 15 + ~2 ms fetch overhead).

**W2 — CP forward batching / fusion.** The CP forward is 15 separate
`forward_one_token_launch/fetch` per frame, each dispatching a 5-layer
transformer forward. Autoregressive: group g+1's input depends on g's
sampled token, so groups **cannot** be batched. But:
- (a) **Speculative decoding**: pre-emit multiple groups assuming a
  greedy argmax path, roll back mismatches. M6.2 tried this and saw
  mixed results (TALKER_SPECULATIVE env var, default off); worth
  re-examining with the current stack.
- (b) **Kernel fusion**: the CP 5-layer transformer decomposes into
  ~12 aclnn ops per layer × 5 layers = 60 dispatches per forward_one_token.
  × 15 groups = 900 dispatches per frame. CannFusion or aclnn fused-op
  equivalents could collapse many of these into fewer kernel launches.
- (c) **Graph capture (aclGraph)**: capture the full forward once, replay
  15 times. M4 tried this for the Talker forward and measured 2.3× slowdown
  on single utt (capture cost not amortized); may be different for CP since
  shape is stable across groups.

W2 is exploratory — **we fund the exploration, not the implementation**,
until one of (a/b/c) shows a credible win estimate.

## 5. Milestones

### W1 — NPU lm_head port (Agent X, ac01)

Ordered; each milestone must land before the next starts.

- [ ] **W1.1** — Load: upload 15 lm_head weights to NPU during
      `CpCannEngine::init_*` (alongside the existing Q/K/V/O/gate/up/down
      uploads). Choose F16 precision (matches what the CP forward path
      already uses). Store as a `std::vector<aclTensor*>` keyed by group.
- [ ] **W1.2** — Dispatch: add `forward_lm_head(group_idx, cp_out_device,
      logits_out_device)`. Input is the `cp_hidden=1024` tensor already
      on NPU (from `forward_one_token`'s last hidden state). Use
      `aclnnMm(logits_out, cp_out_F16, lm_head_W_F16)`. Output is
      `vocab_size=2048` F16 on NPU.
- [ ] **W1.3** — Fetch API: `fetch_logits(group_idx, float* host_out)`
      copies the F16 logits tensor from NPU → CPU, casting to F32 on the
      fly (F16→F32 upconvert is a no-op cost on host). Replaces the
      CPU-side `cp_matvec_f32(cp_f32_.lm_head_w[g], ...)` call in
      `talker.cpp::predict_code_groups`.
- [ ] **W1.4** — Wire + correctness check: in `predict_code_groups`,
      replace `cp_matvec_f32` with the NPU path. Compare first-10-frame
      sampled tokens against the CPU baseline for 3 canonical utts,
      allow ≤ 1 token drift per frame (F16 precision), flag larger drift
      as a correctness bug.
- [ ] **W1.5** — fps measurement: `TALKER_CP_PROF=1` run on canonical
      xvec mayun. Target: lm phase drops from ~15 ms → ≤ 5 ms. Gate the
      whole W1 behind env var `TALKER_LM_HEAD_NPU=1` for A/B.
- [ ] **W1.6** — User-ear check: long-utt xvec mayun with W1 on vs off.
      Audio should be audibly identical; if not, W1.4 correctness gate
      has a leak — investigate.

**Open issues for W1:**
1. Should we quantize lm_head to INT8 (W8-style)? +memory savings, ~same
   fps win. Defer to W1.7 if baseline W1 lands clean.
2. cp_cann_engine currently stores forward hidden state on NPU or in
   CPU-host buffer? If it's already on NPU, W1.3 is a pure-device op
   (no fetch). If host, we need to add an NPU-side buffer for cp_out.
   Agent X to probe and decide.
3. The 5-frame warmup pre-compute happens in `QwenTTS::load()` —
   verify it still triggers lm_head weight upload correctly. (Warmup
   was added in `9b293375` with a fake zero speaker embedding.)

**W1 acceptance**: lm phase ≤ 5 ms/frame (was 15 ms), overall fps ≥ 25
on canonical, user-ear identical on mayun xvec long-utt.

### W2 — CP forward exploration (Agent Y, ac02)

Pure research track — Agent Y measures and reports. No code to merge
until W1 is done and a winner is picked.

- [ ] **W2.1** — Baseline rigour: three 3-trial measurements on
      canonical ICL+xvec+CV to pin the current CP forward wall time.
      Current claim is 17 ms/frame; verify on fresh runs with
      `TALKER_CP_PROF=1`.
- [ ] **W2.2** — Speculative re-audit: re-run `TALKER_SPECULATIVE=1`
      on canonical with current W8+TQE=2+cp_groups=15 stack. Compare
      fps + token-match rate against sequential baseline. M6.2 reported
      mixed results; new stack may behave differently.
- [ ] **W2.3** — aclGraph capture cost model: measure the
      per-capture-plus-replay cost for a single CP forward_one_token.
      If capture is < 1 ms and replay saves > 5 ms per-group, the net
      win per frame is 15 × (save - capture_amortized). Agent reports
      the per-shape curve, not a patch.
- [ ] **W2.4** — Kernel-count audit: count actual aclnn dispatches per
      CP forward_one_token (use `ACL_PROFILING=1` or `nsight` trace if
      available, else instrument `cp_cann_symbols.cpp` with a counter).
      Find the top-3 highest-cost kernels. CannFusion candidate list
      emerges from this.
- [ ] **W2.5** — Report: written analysis at
      `docs/cp_forward_opt_exploration.md`: for each of (a/b/c), say
      "won't work because X", "worth N days for M ms/frame" or
      "blocked on CANN feature Y". PM signs off on which (if any)
      graduates to a W3 implementation milestone.

**Open issues for W2:**
1. `TALKER_SPECULATIVE` was removed (commit `84af6590` defaulted to
   OFF). Is the code still reachable? Agent Y to check before W2.2.
2. aclGraph capture was tried for Talker forward and regressed. CP
   forward is smaller (5 layers vs 28) — capture cost may be
   proportionally smaller. Different shape assumption.
3. Does the native CP engine already batch across timesteps? Re-check
   `cp_cann_engine.cpp` — if group-g and group-g+1 forward share KV
   cache, some overlap may already be exploited.

**W2 acceptance**: the written report. No code landed unless W2.5
recommends it.

## 6. Acceptance criteria (summary)

- [ ] W1 lands, commits `feat(tts): W1 — NPU lm_head port` on fork.
- [ ] Canonical xvec mayun long-utt measures ≥ 25 fps with W1 on.
- [ ] User ear-check confirms identical audio W1 on vs off.
- [ ] W2 report published; PM decides whether to fund a W3
      implementation.
- [ ] If both W1 + best W2 candidate land, canonical ≥ 28 fps.

## 7. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| F16 lm_head drifts sampled tokens on edge cases (sampling is non-monotonic — a near-tie flipping on F16 rounding) | Medium. Causes rare phoneme glitches. | W1.4 correctness gate with ≤1 token drift per frame; gate W1 behind env var so we can revert instantly. |
| `aclnnMm` on 1×1024 @ 2048×1024 is mis-tuned (NPU optimized for larger matmuls) | Medium. Could get < 5 ms savings. | W1.5 measures reality before committing the design. |
| Thread/stream contention between W1's new aclnn dispatches and the existing CP forward | High. Could tank perf on the very path it's supposed to speed up. | Share the CP engine's existing stream for W1 dispatches (already serialized); don't introduce new streams. |
| Speculative decoding (W2.2) re-introduces quality regressions | Medium. | Gate behind env var; PM reviews user-ear output before any default flip. |
| Agent drift on ac02 affects ac01 (same physical host, shared NPU) | Low but real. | W1 uses ac01; W2 uses ac02. Each has its own container + build tree. |

## 8. Parallelism playbook

**Agent X** (ac01, Rust/C++) — executes W1 sequentially (W1.1 → W1.6).
Deliverables: code commits on fork, user-ear confirmation on mayun
xvec, fps numbers.

**Agent Y** (ac02, research) — executes W2 in parallel with W1.
Deliverable: single markdown report at
`docs/cp_forward_opt_exploration.md`. No code in this phase.

**PM (this session)** — arbitrates blockers, reviews user-ear
checks, decides whether to fund W3 based on W2.5.

**Host split (enforced)**:
- ac01: W1 work only. Do not run experiments that touch lm_head from
  ac02.
- ac02: W2 exploration only. Do not run benchmarks that mutate ac01's
  source tree.
- Each agent seeds its own `~/work/OminiX-Ascend-w1/` or
  `~/work/OminiX-Ascend-w2/` to avoid collision with the existing
  `~/work/OminiX-Ascend/` tree.

## 9. Sign-off

- [ ] User signs this contract (PM to request before spawning agents).
- [ ] On sign-off, PM spawns Agent X + Agent Y in parallel.
- [ ] PM reports back per milestone; does not dive into the engineering.

---

**Current status**: draft, pending user sign-off. PM (Claude) will not
dispatch agents until the user marks this contract accepted.
