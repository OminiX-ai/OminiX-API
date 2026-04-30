# Ascend QIE Native Engine — Abstraction-Break Survey

**Status:** Read-only survey. Recommendations are post-§5.5.66 (saga close gates everything below).
**TL;DR:** The native QIE engine at `tools/qwen_image_edit/native/image_diffusion_engine.{cpp,h}` is **already fully native** — no ggml-cann involvement on the forward path. The 30× TTS arc was *getting to native*; QIE is already there. The remaining payoff comes from layering **the same six TTS post-native optimizations** that QIE has not yet adopted: NZ weights, AddRmsNorm fusion, FFNV3, GroupedMatmulV3, aclGraph capture, and TASK_QUEUE_ENABLE=2 pipelining. Realistic ceiling: **~2× over the current native baseline**, not another 30×.

---

## 1. Coverage map

The native engine compiles to 5,647 LOC of direct aclnn dispatch. **`forward_block_` (cpp:3037) does not call any ggml-cann path.** The 60-block DiT runs entirely on the engine's own scratch buffers:

| Op family | Native dispatch (current) | Reference: TTS native equivalent |
|---|---|---|
| Q/K/V projections (6 per block) | 6× `aclnnWeightQuantBatchMatmulV3` (Q4) or 6× `aclnnMm` (F16) | TTS uses fused `aclnnGroupedMatmulV3` (3-into-1) |
| Modulation linears (img_mod, txt_mod) | 2× WQBMMv3 / aclnnMm | same pattern |
| LayerNorm1/2 + modulate | F32 `aclnnLayerNorm` → cast → `aclnnMul` + `aclnnAdd` | (no analogue — TTS uses RMSNorm only) |
| Q/K RMSNorm (4 per block) | 4× `aclnnRmsNorm` (F16 in/out, F32 gamma) | TTS uses `aclnnAddRmsNorm` (fused with prior residual add) |
| RoPE | on-device 4×Mul + 2×Add manual (cpp:2858) or `aclnnRotaryPositionEmbedding` | TTS uses `aclnnApplyRotaryPosEmbV2` (fused 2-call) |
| Joint attention | `aclnnFusedInferAttentionScoreV2` BSND seq=4352 | same op |
| Attn output projections | 2× WQBMMv3 / aclnnMm | same |
| FFN per stream (up + GELU + down) | 3 dispatches (`Mm` → `GeluV2` → `Mm`) | TTS uses fused `aclnnFFNV3` (1 call) |
| Gated residual add (F32 accumulator) | `aclnnMul` + `aclnnCast` + `aclnnInplaceAdd` | TTS uses `aclnnInplaceAddRmsNorm` (fused into next-block RMSNorm) |

**Memory:** scratch buffers are pre-sized at init for the worst-case `max_img_seq + max_txt_seq` (4352) × hidden (3072). Workspace grows lazily. RoPE tables are pre-computed (Q0.5.3). Q4-resident weights (~5.1 GiB) + F16 scales (~0.6 GiB) keep ~18 GiB resident, well under 32 GB HBM. The memory pattern already matches TTS's "workspace tensor reuse" guidance.

**Streams:** single `compute_stream_ = primary_stream_` (cpp:733-734). No dual-stream overlap.

**`ggml-cann` involvement on the hot path:** **zero.** GGUF parsing at `init_from_gguf` uses ggml weight-tensor I/O; that is one-time, not per-step.

---

## 2. What's NOT yet native (vs TTS)

Confirmed by symbol-name grep (`aclnnAddRmsNorm`, `aclnnFFNV3`, `aclnnGroupedMatmul`, `aclnnTransMatmulWeight`, `aclgraph`, `TASK_QUEUE_ENABLE`):

| TTS optimization | TTS impact | QIE state | Source |
|---|---|---|---|
| FRACTAL_NZ weight pre-conversion | +15% (29.7→25.9 was ND→NZ in the writeup, table at qwen_tts_optimization_writeup.md:133-134) | **Not adopted.** No `aclnnTransMatmulWeight` calls. Weights stay ND. | docs/qwen_tts_optimization_writeup.md:127-145 |
| `aclnnFFNV3` (fused up/act/down) | shipped | **Not adopted.** QIE FFN is 3 separate dispatches per stream (cpp:3777-3811). | tools/qwen_tts/cp_cann_engine.cpp:732-921 |
| `aclnnGroupedMatmulV3` for Q/K/V | shipped (A4c Phase 1) | **Not adopted.** QIE does 3 separate matmuls per stream × 2 streams = 6 per block (cpp:3410-3428). | tools/qwen_tts/talker_cann_engine.cpp:660-769 |
| `aclnnInplaceAddRmsNorm` (fuses prior residual add into next RMSNorm) | shipped (Phase A.1) | **Not directly applicable** — QIE uses LayerNorm not RMSNorm at residual sites. But `aclnnAddLayerNorm` (if it exists in CANN 8.5) is the analogue. | tools/qwen_tts/cp_cann_engine.cpp:1495-1526 |
| aclGraph capture | parked in TTS (2.3× *slower* on iterative decode) | **Likely a win for QIE** because QIE is steady-state (60 blocks × 20 steps with constant shapes), not iterative-with-shape-flip. | docs/qwen_tts_optimization_writeup.md:114-128 |
| `TASK_QUEUE_ENABLE=2` | **the final enabler** in TTS — pipelines launch with execute | **Not adopted.** No env-var setup in QIE init. | docs/qwen_tts_optimization_writeup.md §"Final enabler" |

Other notable potentials surfaced by the source:
- **Transient tensor descriptors:** `forward_block_` constructs 76 `aclCreateTensor` / `aclDestroyTensor` pairs per block per step. TTS engine constructs 137 across the *whole* loop with descriptor reuse. At 60 blocks × 20 steps, QIE creates ~91k tensors per image; TTS-style descriptor pooling would amortize this.
- **Modulation broadcast `aclnnMul`:** in `modulate_` the F16 `[B,H]` shift/scale is applied row-wise to `[B,seq,H]` via repeated mul/add — could collapse into a single fused kernel.

---

## 3. Top 5 abstraction-break candidates (ranked by payoff × tractability)

Ranking model: payoff scaled to TTS's deltas, divided by effort. All deltas refer to the *current native baseline* (denoise_full wall-clock per step), not the 1-fps ggml-cann starting point.

### #1 — TASK_QUEUE_ENABLE=2 + aclGraph capture per-block
- **Replaces:** synchronous launch model (host-wait per dispatch)
- **Op count:** ~one-line env set + ~150 LOC of capture/replay machinery copying TTS's `capture_aclgraph_forwards_` (cp_cann_engine.cpp:1672-1677)
- **Expected win:** **medium-large** — TTS reports TASK_QUEUE_ENABLE as the "final enabler"; aclGraph captures 60 blocks × 20 = 1200 replays of identical shape, exactly the workload it was designed for (QIE is steady-state, unlike TTS's iterative decode where graph capture lost). Ballpark +20-40%.
- **Risk:** low numerical (replay is bit-identical), medium correctness (block 0 dump infrastructure breaks under capture; gate behind `QIE_ACLGRAPH=1`)
- **Effort:** ~1 week — symbol surface already loaded in `cp_cann_symbols.h:486-540`

### #2 — `aclnnFFNV3` for img / txt FFN
- **Replaces:** 6 dispatches per block (3× img + 3× txt: up Mm + GeluV2 + down Mm) → 2 dispatches
- **Caveat:** QIE FFN is GELU-tanh, not SwiGLU. `aclnnFFNV3` accepts `activation="gelu"` per the symbol header. Verify on CANN 8.5 sample list.
- **Expected win:** **medium** — TTS's FFN is the largest single sublayer; same is true for QIE (FF=12288 = 4× hidden). Ballpark +10-15%.
- **Risk:** low if CANN 8.5 supports gelu in FFNV3; if not, the alternative is `aclnnFusedActivationLinearForward` or a custom AscendC gelu+matmul kernel (effort jumps 5×).
- **Effort:** ~3 days if FFNV3-gelu works, else 2-3 weeks for AscendC.

### #3 — `aclnnGroupedMatmulV3` for Q/K/V
- **Replaces:** 3 separate WQBMMv3 / aclnnMm calls per stream per block → 1 grouped dispatch
- **Caveat:** TTS's GroupedMatmulV3 is gated on W8 (INT8) weights. QIE uses Q4_K antiquant via WQBMMv3. Need to check whether GroupedMatmulV3 has a WQBMM-grouped sibling or whether Q/K/V need to be batched at the WQBMM level via stride packing.
- **Expected win:** **small-medium** — TTS reports GMM at ~94-100 μs vs ~98-102 μs for 3× WQBMMv3 (4-8% per attention sublayer; symbols header notes this at line 435). Ballpark +5-8% if portable.
- **Risk:** medium — if no Q4 grouped variant exists, this is a wash on QIE.
- **Effort:** ~1 week including the bias-channel-pack rework.

### #4 — FRACTAL_NZ weight pre-conversion at init
- **Replaces:** ND-format weight at every WQBMMv3 / aclnnMm call → NZ-format (Ascend's native cube tiling)
- **Expected win:** **small-medium for Q4 path, medium for F16 fallback path** — TTS measured +15% in the F16 path (writeup §"Layer 3"). The Q4 WQBMMv3 path may or may not benefit; CANN docs (`aclnn_weight_quant_batch_matmul_v3.h`) need verification on whether mat2 NZ is consumed.
- **Risk:** low — TTS proved it; metadata mutation is in-place.
- **Effort:** ~3 days — copy `set_use_nz_weights` from `cp_cann_engine.cpp:424-462` and apply at every Linear weight in `init_from_gguf`.
- **Caveat:** this is the lowest-risk win listed but the smallest expected magnitude.

### #5 — Tensor descriptor pool + AscendC custom modulate kernel
- **Replaces:** 91k transient `aclCreateTensor`/`aclDestroyTensor` per image; the 4-op `modulate_` (Mul + Add + Cast chain) with one fused kernel
- **Expected win:** **small** — descriptor work is already inside a single host thread; CANN's tensor descriptor allocator is fast. Likely +2-5%.
- **Risk:** low for descriptor pool; AscendC kernel is high effort.
- **Effort:** descriptor pool ~3 days; AscendC modulate ~2-3 weeks. **Skip the AscendC half** — won't pay back.

### Honorable mention — Drop the F32 residual stream once §5.5.66 closes
- The Q2.4.5.4d "QIE_ALL_BF16" path (cpp:3094-3102) was the §5.5 saga's overflow workaround. Once the dtype audit closes (§5.5.66) the residual can return to F16, halving residual bandwidth and removing the F32→F16 cast pair per residual add. Estimated +5-8%, but it's *coupled* to saga close, so it doesn't survey separately.

---

## 4. Recommendation (post-§5.5.66)

**Attack #1 first: TASK_QUEUE_ENABLE=2 + per-block aclGraph capture.**

Reasons:
1. Highest expected payoff per LOC. The TTS writeup calls `TASK_QUEUE_ENABLE=2` "the final enabler"; pairing it with aclGraph (which TTS *parked* because TTS's iterative shape-flip kills it but QIE's identical 60-block steady-state matches it) is the unique QIE win. The combination is exactly the "pre-warmed caches for repeated shapes" pattern from the survey prompt.
2. Numerically free. Graph replay is bit-identical to eager dispatch; no precision concerns to bisect against the §5.5 evidence pack.
3. Decoupled from the saga. It does not depend on residual-stream dtype, RoPE layout, or any of the §5.5 evidence chain. Can run on any post-§5.5.66 baseline.
4. Symbol surface already in the dlsym shim (`cp_cann_symbols.h:486-540`).
5. Compounds with #2-#4. Once aclGraph captures the per-block sequence, swapping FFNV3 / GroupedMatmul / NZ inside that captured graph is mechanical (capture-replace-recapture).

Targets: realistic stretch is **~2× current native** with all five layers landed (TASK_QUEUE +25%, FFNV3 +15%, GMM +6%, NZ +12%, descriptor pool +3% — multiplicative ≈ 1.75-2.0×). Not another 30×; the engine's already past the easy gains the TTS arc captured.

---

## File index
- `/Users/yuechen/home/OminiX-Ascend/tools/qwen_image_edit/native/image_diffusion_engine.h` (870 LOC)
- `/Users/yuechen/home/OminiX-Ascend/tools/qwen_image_edit/native/image_diffusion_engine.cpp` (5,647 LOC)
- `/Users/yuechen/home/OminiX-Ascend/tools/qwen_image_edit/native/main_native.cpp` (152 LOC, lock + driver)
- `/Users/yuechen/home/OminiX-Ascend/tools/qwen_tts/cp_cann_symbols.{h,cpp}` (978 LOC, dlsym shim used by both engines)
- `/Users/yuechen/home/OminiX-Ascend/tools/qwen_tts/cp_cann_engine.cpp` (3,674 LOC, reference for NZ + aclGraph + AddRmsNorm patterns)
- `/Users/yuechen/home/OminiX-Ascend/tools/qwen_tts/talker_cann_engine.cpp` (reference for GroupedMatmulV3 and FFNV3 wiring)
- `/Users/yuechen/home/OminiX-Ascend/tools/ominix_diffusion/src/qwen_image.hpp` (847 LOC, CPU reference for parity)
- `/Users/yuechen/home/OminiX-Ascend/docs/qwen_tts_optimization_writeup.md` (the 30× TTS retrospective; baseline-and-target numbers, NZ +15%, W8 +14%)
