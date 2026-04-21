# aclGraph Capture-Replay — Qwen3-TTS CP Forward Dispatch-Floor Elimination

## 1. Status & mandate

**Status**: NEW (drafted 2026-04-21, PM signed). G0 feasibility probe
PASS (see `docs/aclgraph_feasibility.md`).

**Ceiling claim to beat**: clean-quality canonical xvec mayun zh
delivers **30.5 fps** (W8 + TQE=2 + cp_groups=15 + sampling + W1 NPU
lm_head + W3b cp-fusion) — tagged `clean-30.5fps-ear-verified` on fork.

**Goal**: **≥ 32 fps** on the same benchmark with identical user-ear
verdict ("no rumble, clean voice"). Realistic upside per G0 probe
math = +6 to +10 fps (30.5 → 36-40 fps), assuming the
`aclmdlRICaptureTaskUpdateBegin/End` primitive accepts FIAv2's `seq_len`
rebind. Fallback (pos-keyed graph cache, ~17 graphs) preserves the
same upside at ~1 extra day of work.

**PM role**: supervise; each workstream is an agent deliverable with a
numerical gate. PM does not write kernel code.

## 2. Background — why aclGraph (over the parked alternatives)

- **Path C (hand-written AscendC)**: paused. W4.1.3 wired but W4.1.4
  drift gate FAILED (max_drift=1949, 1/256 exact-match). Kernel tile /
  rootcause work is 2-3 weeks for uncertain fps — vendor
  `aclnnFusedInferAttentionScoreV2` is already 40-core-tiled.
- **CannFusion (codegen)**: parked at F0 PASS; §1.12 upstream CANN
  runtime bug blocks fused-epilogue dispatch (`ACL_ERROR_RT_AICORE_EXCEPTION
  507015`). Open after 4 fix waves; no ETA.
- **aclGraph**: API present on 8.3.RC1 (38 `aclmdlRI*` symbols resolve
  from `libascendcl.so`). Production reference exists (vllm-ascend ACL
  Graph path uses the identical task-update primitive for paged
  attention rebinding). CP `forward_one_token` has exactly 3
  parameter-update points (RoPE cos/sin slice, KV-slot write offset,
  FIAv2 `seq_len`); everything else is shape-stable. Capture-once,
  replay-per-frame collapses remaining host-dispatch overhead +
  inter-op scheduling gaps that TQE=2 cannot merge.

**Honest math** (G0 corrected the contract's initial thesis): TQE=2
already amortized per-dispatch launch from ~40 μs (eager) to ~2-3 μs
(two-phase submit). aclGraph does not kill the ~10× dispatch floor
that eager-mode analysis implied — it recovers the residual ~0.5-1.5
ms/forward of host overhead + pipeline gaps. At 15 forwards/frame and
~33 ms/frame baseline, that's +6 to +10 fps.

## 3. Scope

**In scope**:
- G1 one-layer capture smoke — HARD GATE on `CaptureTaskUpdate` semantic.
- G2 full `forward_one_token` capture with per-pos rebinding.
- G3 parity gate: token drift ≤ 1 per frame per group on canonical
  xvec mayun zh at `--cp_greedy`. Must be byte-identical preferred.
- G4 perf HARD GATE ≥ 32 fps + user-ear gate.
- `TALKER_CP_ACLGRAPH=1` env gate. Default off. No behavioural change
  when unset.

**Out of scope**:
- Talker (non-CP) transformer capture — separate contract if ever
  pursued.
- Session-aclGraph (persist across utterances) — W3a territory;
  deferred until FFI bridge session API lands.
- Multi-utterance re-capture cost optimisation — capture at engine
  init, amortise across utterances via existing `QwenTtsCtx` lifetime.

## 4. Host plan

- **Primary**: `ac01` (port 31984), CANN 8.3.RC1, Ascend 910B4.
  Working dir `~/work/OminiX-Ascend-w1/` (same as prior tracks).
- **Secondary**: `ac02` / `ac03` available for parallel perf-bench runs.
- **Fork**: `https://github.com/ymote/OminiX-Ascend` `main`. Path-C
  lane (`3ac9aab5`) stays landed env-gated off. ac01 has no git push
  creds → same patch mechanism as Path C: `git format-patch` on ac01 +
  scp to Mac + PM pushes.
- All diagnostics on Mac (no CANN headers locally) are expected noise.

## 5. Workstreams with gates

### G0 — Feasibility probe (DONE, PASS)

- [x] G0.1 API presence on 8.3.RC1. **PASS** — 38 `aclmdlRI*` symbols
      resolve, header at `acl/acl_rt.h` lines 269-309 + 3210-3393.
- [x] G0.2 Semantic match vs CP forward. **PASS** — vllm-ascend
      (https://docs.vllm.ai/projects/ascend/.../ACL_Graph.html) uses
      `CaptureTaskUpdate` for paged-attn rebinding; semantic maps 1:1
      to our 3 param-update points.
- [x] G0.3 Engine fit audit. **PASS** — 96-op forward inventory, 0
      non-capturable ops, 3 param-update classes all addressable.
- [x] G0.4 Verdict: **CONDITIONAL-GO** on `CaptureTaskUpdate`
      accepting FIAv2 `seq_len` rebind (G1 resolves). Effort 4-6 days
      to runnable first-cut.

Verified-by Agent G0 (2026-04-21): `docs/aclgraph_feasibility.md`.

### G1 — One-layer capture smoke (DONE, YELLOW verdict)

**Verified-by Agent G1 (2026-04-21, fork commit `20ed1871`)**.

- [x] G1.1 Extend `cp_cann_symbols.{h,cpp}` + optional dlsym. **PASS**
      — all 4 task-update symbols resolve on 8.3.RC1
      (`aclmdlRICaptureTaskGrpBegin/End`,
      `aclmdlRICaptureTaskUpdateBegin/End`).
      `has_aclgraph_task_update()` returns true.
- [x] G1.2 One-layer capture harness (`test_aclgraph_smoke.cpp`,
      1046 LoC, synthetic weights, gated by
      `-DQWEN_TTS_ACLGRAPH_SMOKE=ON`). Capture at `pos=0` with FIAv2
      + RoPE + KV-slot each in its own `TaskGrp`. Capture success.
- [x] G1.3 Replay 10× with pos 0..9 and per-op TaskUpdate rebind.
      Per-op results:
        RoPE-Q rebind:    **PASS**
        RoPE-K rebind:    **PASS**
        FIAv2 seq_len:    **PASS** ← the critical question
        KV-slot memcpy:   **PASS_EXTERNAL** — `aclrtMemcpyAsync` D2D
                          NOT CAPTURABLE on 8.3.RC1 (err 507009
                          "task not supported"). Must launch outside
                          captured region.
      Parity multi-grp: **max_abs_diff = 3e-3 F16** (gate was 1e-4 —
      30× off). Likely harness stream-barrier bug, not a driver bug.
      Per-replay wall (ms): [0.332, 0.347, 0.301, 0.303, 0.308, 0.303,
      0.302, 0.300, 0.305, 0.302], **median 0.30 ms / layer** —
      PASSES 1.5 ms gate by 5×. Extrapolates to ~1.5 ms / 5-layer
      forward vs current ~18 ms eager → **projected +10 fps**.
- [x] G1.4 Verdict: **YELLOW**. Task-update semantic works for every
      parameter class we care about (including FIAv2 seq_len). Replay
      timing blows past the gate. The 3e-3 parity drift is the only
      open concern and is not debuggable within G1 scope.

**Hard constraints surfaced for G2**:
1. **One TaskGrp per op**: driver rejects multi-op TaskGrp updates
   ("total=3 success=1 failed=2", `rtsStreamEndTaskUpdate task group
   update error`). Required: one `TaskGrpBegin/End` per rebound op.
2. **No D2D memcpys inside capture**: V→KV-slot write and any
   residual-copy memcpys must launch on `main_stream` BEFORE
   `aclmdlRIExecuteAsync`. Design around this.

**PM decision (2026-04-21)**: proceed to G2 with **Option 1 —
pos-keyed graph cache**. Rationale:
- Avoids the 3e-3 multi-grp parity drift that single-graph TaskUpdate
  would force us to debug.
- Minimal-risk surface: only uses the all-green subset (`CaptureBegin/
  End` + `ExecuteAsync`, no TaskUpdate).
- 17 graphs × ~1 MB = ~17 MB RAM; trivial.
- Captured per-pos graphs bake `seq_len` into the executor so no
  rebind needed at all. Simplest possible production architecture.
- Same projected fps upside (+6 to +10 fps).

Option 2 (single graph + per-op TaskUpdate) remains available as a
future optimisation if fps ceiling needs another push. Deferred.

### G1-legacy — single-graph + TaskUpdate (DEFERRED)

Option 2 is not in active scope. If revisited, first step is debugging
the 3e-3 parity drift in the harness at
`tools/qwen_tts/test_aclgraph_smoke.cpp` (likely: missing
`aclrtStreamWaitEvent` between the pure-head output (main stream) and
the rebound-capture region consuming it).

### G2 — Full `forward_one_token` capture, pos-keyed cache (DONE)

**Verified-by Agent G2 (2026-04-21, fork commit `63a3d90e`)**.

- [x] G2.1 Buffer design. Engine-owned fixed dev buffers; per-pos tensor
      views reference same buffers with pos-specific offsets. No
      per-frame pointer rebind. **PASS**.
- [x] G2.2 Capture-at-init. 17 warmup forwards captured, each into an
      `aclmdlRI` handle via `aclmdlRICaptureBegin(GLOBAL)`. **17/17
      captures succeeded** on first run (RELAXED fallback not needed).
- [x] G2.3 D2D memcpy audit + refactor. 3 D2D memcpys/layer × 5 layers
      = 15/forward moved outside captured region: V→v_cache_slot
      replaced by V-proj descriptor pointing at `v_cache_dev_[il] +
      pos*kv_dim` (baked per-graph); `residual = cur` replaced by
      `aclnnAdd(cur, zero_f16_cp_dev_, alpha=1, residual)` with new
      engine-owned zero buffer.
- [x] G2.4 Per-frame replay path. `forward_one_token_launch` branches
      on `cp_aclgraph_applied_`; fixed-buffer H2D/D2H around
      `aclmdlRIExecuteAsync(graphs_[pos], main_stream)`.
- [x] G2.5 Re-capture semantics. Capture once at engine init;
      `aclmdlRIDestroy` in destructor. Unset env = stock path,
      bit-identical.

**Scope guard**: capture only runs under canonical path
`cp_fusion_applied_ && w8_applied_ && !cp_ascendc_applied_`; other
combos skip capture silently.

**Smoke**: 100-frame canonical xvec mayun zh completes both
`TALKER_CP_ACLGRAPH=1` and unset, no crash.

### G2-footnote — (original spec text preserved for reference)

Per PM G1 decision: **Option 1 — pos-keyed graph cache**. Captures
17 full-forward graphs at engine init, one per `pos ∈ [0, 16]`.
Per-frame dispatch = pointer-select + `aclmdlRIExecuteAsync`.

- [ ] G2.1 Buffer design: engine-owned fixed dev buffers for input
      (H2D-staged), output, residual/normed/cur scratches. All graph
      I/O points to these fixed pointers — no per-frame pointer
      rebind needed, eliminates TaskUpdate path entirely.
- [ ] G2.2 Capture-at-init path: extend `CpCannEngine::init(...)`
      post-weight-upload. If `TALKER_CP_ACLGRAPH=1` and
      `has_aclgraph()`, run 17 warmup forwards in series, each with
      `aclmdlRICaptureBegin(stream, GLOBAL) → forward_one_token
      → aclmdlRICaptureEnd`. Store the 17 `aclmdlRI` handles in a
      `std::vector<aclmdlRI>`. If capture of any pos fails, disable
      aclgraph silently and fall back to eager.
- [ ] G2.3 D2D memcpy audit: identify every D2D `aclrtMemcpyAsync`
      inside `forward_one_token_launch` (V-to-KV-slot,
      `residual = cur`, `cur = residual`, etc.). Launch them on
      `main_stream` BEFORE `aclmdlRIExecuteAsync` — NOT inside the
      captured region. Verify capture doesn't record them (capture
      will silently accept but replay will break — G1 found this).
- [ ] G2.4 Per-frame replay path: `forward_one_token_launch` branches
      on `cp_aclgraph_applied_`. If true:
        (a) H2D-memcpy input to fixed input buffer (outside capture)
        (b) Launch all pre-capture D2D memcpys on main_stream
        (c) `aclmdlRIExecuteAsync(graphs_[pos], main_stream)`
        (d) Read output from fixed output buffer
      Otherwise stock eager path.
- [ ] G2.5 Re-capture semantics for fresh `QwenTtsCtx`: capture is
      engine-lifetime, not utt-lifetime. Graphs persist across
      consecutive TTS calls. Verified by running 3 sequential calls
      with `TALKER_CP_ACLGRAPH=1` and checking no crash / no graph
      leak.

**Gate**: runs without crash on canonical mayun xvec zh, 100 frames,
env `TASK_QUEUE_ENABLE=2 TALKER_W8_QUANT=1 TALKER_LM_HEAD_NPU=1
TALKER_CP_FUSION=1 TALKER_CP_ACLGRAPH=1`.

**Risk**: max-pos needs to be large enough. Canonical benchmarks use
cp_groups=15, so pos in [0, 14]. Set MAX_ACLGRAPH_POS=16 for margin.
If a longer sequence somehow exceeds, fall back to eager for positions
beyond cache; graph cache only covers common path.

### G3 — Correctness parity gate (DONE)

**Verified-by Agent G2 (2026-04-21, fork commit `63a3d90e`)**.

- [x] G3.1 Canonical mayun xvec zh `--cp_greedy --seed 42`.
      Per-frame max-drift first 16 frames: **all zeros**.
      max_drift_over_240 = **0**. 100-frame extended dump (1680
      tokens) max drift = **0**, sum = 0. Output wav md5 identical.
      **Byte-identical parity. PASS by perfect margin.**
- [x] G3.2 ICL mayun zh + CV serena zh regression runs:
      - ICL: token diff exit=0, wav md5 `fbb67b02dfb13b0432026e06c059a4ed`
        matches stock
      - CV: token diff exit=0, wav md5 `d68633035e4621e3737998fd07e5a1e3`
        matches stock
      Both byte-identical, aclgraph-off path untouched.
- [x] G3.3 No debug needed — G3.1 passed on first run.

**Gate**: **PASS** byte-identical across xvec / ICL / CV.

### G3-sanity — perf preview (not a gate, G4 is authoritative)

`TALKER_CP_PROF=1` 50-frame xvec mayun zh:
- Stock median `fwd = 18.34 ms` (range 18.3-19.1)
- aclgraph median `fwd = 17.83 ms` (range 17.8-17.84)
- Per-forward delta: ~0.5 ms (at lower bound of G1's extrapolation)

xvec wall 43 frames: 1340.8 ms stock → 1289.1 ms aclgraph (~4% wall
win). Projected: 30.5 fps → ~31.7 fps wall on this short test. Full
canonical run (G4) likely higher due to capture amortisation.

### G4 — Perf + user-ear HARD GATE (1 day)

- [ ] G4.1 `TALKER_CP_PROF=1` on ac01: full-run `talker_xvec` wall
      with / without `TALKER_CP_ACLGRAPH=1`. Canonical benchmark,
      3 runs, take median.
- [ ] G4.2 Gate: **fps ≥ 32** (goal) with stretch target ≥ 36.
      `forward_one_token` wall drops from ~18 ms to ≤ 14 ms.
- [ ] G4.3 User-ear gate: deliver two wavs to
      `/Users/yuechen/Downloads/tts_ab/`:
        `g4_pre.wav`  — `TALKER_CP_ACLGRAPH=0` (= 30.5 fps baseline)
        `g4_final.wav`— `TALKER_CP_ACLGRAPH=1`
      PM listens. Any rumble / distortion / new artefact = REJECT.

**Gate**: fps ≥ 32 AND user-ear clean. If either fails, roll env to
unset-default, document outcome, close contract.

**On PASS**: commit `TALKER_CP_ACLGRAPH` env gate, author
coverage-sweep wavs (icl/cv/xvec/zh/en), tag
`aclgraph-{fps}-ear-verified`.

## 6. Acceptance criteria (summary)

- [x] G0 probe PASS → proceed.
- [ ] G1 HARD GATE: task-update accepts FIAv2 rebind OR fallback
      pos-keyed cache viable, 1-layer replay ≤ 1.5 ms.
- [ ] G2 full-forward capture lands env-gated on fork.
- [ ] G3 parity drift ≤ 1 on canonical xvec mayun zh.
- [ ] G4 fps ≥ 32 + user-ear clean.
- [ ] Contract §5 all `[x]` with `Verified-by` stamps + commit SHAs.

## 7. Risks

1. **Task-update rejects FIAv2 `seq_len`** — addressed by fallback
   (pos-keyed cache, +1 day). Real risk level: low.
2. **Replay wall ≥ eager wall** — would be a structural failure of the
   thesis. Unlikely given vllm-ascend production evidence but not
   ruled out until G1.4 measures. If observed: abandon, report root
   cause, consider group-collapse algorithmic track instead.
3. **Stream/event races on W1 lm_head wait** — mitigated by `GLOBAL`
   capture mode. Fallback: record the wait outside the captured range.
4. **Silent numerical drift** — F16 rounding could accumulate
   differently under replay vs eager. Mitigated by G3.1 byte-drift
   gate and G4.3 user-ear gate. If drift is small-but-non-zero:
   PM arbitrates whether to ship.
5. **Re-capture cost on engine restart** — capture happens once at
   init via the existing warmup forward; no per-utt re-capture. Verified
   by G2.3.
6. **Task-update workspace conflict on shared streams** — the
   `CaptureTaskUpdate` side-stream pattern requires a separate
   `aclrtStream` for the update op. Create + dispose in G2 setup.

## 8. Host rules (carried from Path C / CannFusion)

- All kernel-facing work on ac01. Mac LSP noise on CANN-dependent files
  is expected (no local headers).
- No force-push to fork `main` without PM authorisation. All merges
  via PM's Mac. ac01 has no git push creds; patch-file mechanism.
- Each gate stops the agent for PM review. No silent continuation past
  a failing gate. G1 and G4 are HARD KILL.
- No Claude coauthor on commits.
- Token-drift and user-ear gates outrank fps numerics. A wrong 40 fps
  ships nothing; a clean 32 fps ships.
