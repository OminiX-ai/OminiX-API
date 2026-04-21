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

### G1 — One-layer capture smoke (2 days, HARD GATE)

**Purpose**: answer the single remaining feasibility question — does
`aclmdlRICaptureTaskUpdateBegin/End` accept re-binding of the FIAv2
`seq_len` int64 scalar + K/V stride-over-seq tensor views on an
already-captured `aclOpExecutor`?

- [ ] G1.1 Extend `cp_cann_symbols.cpp` to optionally resolve the
      task-group + task-update symbols: `aclmdlRICaptureTaskGrpBegin`,
      `aclmdlRICaptureTaskGrpEnd`, `aclmdlRICaptureTaskUpdateBegin`,
      `aclmdlRICaptureTaskUpdateEnd`. Add a `has_aclgraph_update()`
      capability probe. If any symbol missing: STOP, report, PM
      decides.
- [ ] G1.2 Isolate one transformer layer (`il=0`) into a standalone
      capture harness inside `cp_cann_engine.cpp` behind
      `#ifdef TALKER_CP_ACLGRAPH_SMOKE`. Capture at `pos=0`, record
      the layer-0 sub-DAG into a `aclmdlRI` handle with FIAv2 +
      RoPE + KV-slot-write inside a `TaskGrp` block.
- [ ] G1.3 Replay 10× with `pos` varying 0..9. Each replay uses
      `CaptureTaskUpdateBegin/End` on a side stream to rebind:
        (a) RoPE cos/sin strided slice pointer
        (b) KV-cache write offset (`k_cache_dev_[0] + pos * kv_dim`)
        (c) FIAv2 `seq_len` scalar
      Verify output bit-identical (or ≤ 1e-4 F16 abs-diff) vs the
      stock eager path for all 10 positions.
- [ ] G1.4 Wall-time per replay vs eager. Record with
      `aclrtRecordEvent` around the replay call and subtract.

**Gate**: parity PASS + **replay ≤ 1.5 ms / layer** (extrapolated to
7.5 ms / forward, vs current ~18 ms). If parity fails OR replay
≥ eager, STOP, report:
- Parity fail → debug; investigate whether `seq_len` is templated into
  the executor (the feared failure mode).
- Slower-than-eager → abandon aclGraph; pivot to group-collapse or
  revisit Path C with proper tiling.

**Fallback path** (pre-approved, same scope): if `CaptureTaskUpdate`
rejects FIAv2 seq_len but replay-at-fixed-pos is fast, switch to
**pos-keyed graph cache** (~17 graphs, one per valid `pos` in 0..16).
Same fps upside; adds ~1 day to G2.

### G2 — Full `forward_one_token` capture (2 days, gated on G1)

- [ ] G2.1 Capture-at-init path: extend `CpCannEngine::init_post_weights_`
      (or equivalent post-weight-upload hook) to run one warmup forward
      and, if `TALKER_CP_ACLGRAPH=1`, capture it into an `aclmdlRI`
      handle. Use `ACL_MODEL_RI_CAPTURE_MODE_GLOBAL` (tolerates
      cross-stream W1 lm_head wait).
- [ ] G2.2 Per-frame replay path: `forward_one_token_launch` branches
      on `cp_aclgraph_applied_`. If true, `aclmdlRICaptureTaskUpdateBegin`
      → rebind 3 param-update points → `aclmdlRIExecuteAsync` →
      `CaptureTaskUpdateEnd`. Otherwise stock path.
- [ ] G2.3 Re-capture semantics for fresh `QwenTtsCtx`: capture is
      engine-lifetime, not utt-lifetime. Verify engine re-use across
      consecutive TTS calls does not leak the captured graph.

**Gate**: runs without crash on canonical mayun xvec zh, 100 frames,
env `TASK_QUEUE_ENABLE=2 TALKER_W8_QUANT=1 TALKER_LM_HEAD_NPU=1
TALKER_CP_FUSION=1 TALKER_CP_ACLGRAPH=1`.

### G3 — Correctness parity gate (1 day)

- [ ] G3.1 Canonical mayun xvec zh `--cp_greedy --seed 42
      --max_tokens 16`. Compare token IDs with `TALKER_CP_ACLGRAPH`
      unset vs set. Gate: ≤ 1 token drift per frame per group for
      first 16 frames (same shape as W1.4 / W4.1.4).
- [ ] G3.2 ICL full run + customvoice run as secondary regression —
      sanity that non-xvec paths are untouched when `TALKER_CP_ACLGRAPH=1`.
- [ ] G3.3 If drift > 1 anywhere, debug before proceeding. Common
      suspects: rebind order (cos/sin vs KV-slot vs seq_len),
      executor reuse across replays, stale workspace.

**Gate**: drift ≤ 1 on xvec; ICL + CV have no regression.

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
