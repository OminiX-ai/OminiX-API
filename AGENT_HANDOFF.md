# Agent Handoff: OminiX Makepad Comparison + Ascend TTS Optimization

## Context

This session started as a deep code review of Makepad's `libs/` vs OminiX-MLX, then evolved into cross-platform TTS optimization spanning three repos and two hardware platforms.

## Repos & Locations

| Repo | Path | Remote |
|------|------|--------|
| OminiX-MLX | `/Users/yuechen/home/OminiX-MLX/` | `oxiglade/mlx-rs` |
| OminiX-API | `/Users/yuechen/home/OminiX-API/` | `OminiX-ai/OminiX-API` |
| OminiX-Ascend | `/Users/yuechen/home/OminiX-Ascend/` | `OminiX-ai/OminiX-Ascend` |
| MFA | `/Users/yuechen/home/OminiX-MLX/universal-metal-flash-attention/` | (submodule) |
| Makepad (reference) | `/tmp/makepad-dev/` | shallow clone, read-only |

## Ascend Server Access

```bash
ssh -i ~/home/tensordock/KeyPair-4fbd-yue.pem -p 31984 ma-user@dev-modelarts.cn-southwest-2.huaweicloud.com
```
- Ascend 910B4, 32GB HBM, CANN 8.3.RC1
- Code deployed at: `~/work/OminiX-Ascend/`
- GGUF models: `~/work/OminiX-Ascend/tools/qwen_tts/gguf/` (Base) and `gguf_cv/` (CustomVoice)
- Always set env before running:
```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$ASCEND_TOOLKIT_HOME/runtime/lib64:$LD_LIBRARY_PATH
source $ASCEND_TOOLKIT_HOME/../set_env.sh 2>/dev/null || true
```

## AutoDL Server Access (China GPU server, for relaying files)

```bash
sshpass -p "hWOBhhnRX0lo" ssh -p 13797 root@connect.gda1.seetacloud.com
```
- Use `source /root/miniconda3/etc/profile.d/conda.sh && conda activate base` for Python
- HuggingFace blocked from both servers. Use local Mac to download, relay via AutoDL.
- AutoDL `/tmp` is FULL (30GB). Use `/root/autodl-tmp/` (82GB free).

## What Was Shipped

### OminiX-MLX (all committed, not pushed)

1. **Memory module** (`mlx-rs-core/src/memory.rs`): `MemoryGuard`, `eval_with_retry`, `memory_snapshot`, `preflight_check`, `clear_cache`, `set_memory_limit`, `set_cache_limit`
2. **MemoryGuard propagated** to 7 crates: mixtral-mlx, qwen3-mlx (3 files), qwen3.5-35B-mlx, gpt-sovits-mlx, qwen-image-mlx. All `unsafe { mlx_clear_cache() }` replaced.
3. **Manual matmul attention → MLX SDPA** in 4 crates: funasr-mlx (encoder+decoder), deepseek-ocr2-mlx (2 files), qwen3-vl-mlx. Measured 1.3-2.2x attention speedup.
4. **Batched QKV projection** in qwen3-tts-mlx: merged q_proj/k_proj/v_proj into single quantized matmul. Clean refactor, speed-neutral.
5. **Flash attention Metal kernel** (`mlx-rs-core/src/metal_kernels.rs`): Online softmax, GQA-native, causal masking. Tests pass. Slower than MLX SDPA for Q=1 decode but ready for ASR/FLUX long prefill.
6. **MFA Metal fix**: Replaced `__asm` AIR intrinsics with barrier-based cooperative copy for macOS 26 compat. Added `MTLCompileOptions` with `.version3_1`.
7. **MFA GQA**: Added `num_kv_heads` to FFI, fixed `batchSize` passthrough, auto-select broadcast mode.

### OminiX-API (committed, not pushed)

1. **GPU memory monitoring**: `memory_snapshot()` logged at model load and per-request for image/llm/tts engines.
2. **eval_with_retry**: Denoising loop + VAE decode in image engine (2 retries on OOM).
3. **Preflight check**: Estimated VRAM per model type, warns before loading.
4. **Safe clear_cache**: Replaced `unsafe { mlx_sys::mlx_clear_cache() }` with `memory::clear_cache()`.

### OminiX-Ascend (committed, not pushed)

Commits (8 total on main):
1. `daf1f33` — Language tolower fix for xvec/customvoice
2. `54b6080` — MRoPE 4x positions, remove double tts_pad, llama.cpp CANN path for xvec (30x speedup: 0.43→12.8 fps)
3. `808a5d6` — Standard RoPE works for TTS (MRoPE GGUF produces noise)
4. `c856721` — CustomVoice mode, speaker ID export, Q8/Q5/Q4 quantization benchmarks
5. `5765e99` — Batch reuse optimization (+15%, 13.0 fps)
6. `dabca7d` — cp_groups configurable flag
7. `61c75ff` — cp_layers flag + native CANN engine prototype
8. `12405a5` — CANN engine build findings, cp_layers test results

## Current Performance Numbers

### Apple M3 Max (MLX, 8-bit quantized)
- Qwen3-TTS: **46 fps, ~3.1x realtime** (cannot be improved further — MLX ceiling)

### Ascend 910B4 (llama.cpp + CANN, Q8_0)
- ICL voice clone: **13.0 fps, 1.48x realtime** (best achieved)
- xvec voice clone: **12.8 fps** (speech quality OK but not as good as MLX)
- CustomVoice vivian: **~12 fps** (working with proper CV GGUF)
- Q5_K_M: 3.8 fps (CANN not optimized for Q5)
- Q4_K_M: 4.5 fps (same issue)

### Bottleneck Analysis (Ascend, per frame at 13 fps = 77ms/frame)
- Talker LLM decode (28 layers, NPU): **17ms** — fast, well-optimized
- Code Predictor (5 layers × 14 decodes, NPU via llama.cpp): **46ms** — bottleneck
  - Each `llama_decode()`: ~3ms (2ms launch overhead + 1ms compute)
  - 14 sequential calls: ~42ms NPU + ~4ms CPU (lm_heads, embedding lookups)
- Codec head + sampling + embedding: **14ms** — CPU, already NEON-optimized

## What Failed (Don't Retry)

| Approach | Why it failed |
|----------|--------------|
| Fused RMSNorm+QKV projection kernel (MLX) | MLX lazy eval already batches — 0% gain |
| Fused QK-norm+RoPE kernel (MLX) | Same — MLX handles dispatch batching |
| Flash attention for TTS decode (MLX) | MLX SDPA faster at Q=1 |
| Custom quantized matmul (MLX) | MLX already fused+M3-tuned, <5% theoretical |
| MRoPE GGUF for TTS (Ascend) | Standard RoPE works; MRoPE produces noise |
| CANN graph mode `GGML_CANN_ACL_GRAPH=on` (Ascend) | NPU core crash on TTS embedding inputs |
| GGML CP session on CANN (Ascend) | Segfault — custom CP ops not supported |
| CPU NEON CP path (Ascend) | Too slow for 5-layer transformer |
| cp_groups < 15 (Ascend) | EOS breaks + quality degrades (model not trained for partial codebooks) |
| cp_layers < 5 (Ascend) | Only layers=5 produces audible speech |

## What To Do Next (Priority Order)

### 1. CANN Engine Integration (HIGH — the only path to 20+ fps)

**Status**: 850-line prototype at `tools/qwen_tts/cp_cann_engine.{h,cpp}`. Code is written. Build fails due to CANN library circular dependencies.

**Fix options** (try in order):
1. **dlsym approach**: Load `aclnnMm`, `aclnnRmsNorm`, `aclnnSilu`, `aclCreateTensor` at runtime via `dlopen("libopapi.so")`. Avoids all link-time dependency issues. ~50 lines of function pointer declarations.
2. **Compile into ggml-cann.so**: Add `cp_cann_engine.cpp` to `ggml/src/ggml-cann/CMakeLists.txt` instead of `tools/qwen_tts/CMakeLists.txt`. All CANN symbols already resolved there.
3. **Separate shared lib**: Build `libcp_cann.so` with proper CANN link flags, dlopen it from qwen_tts.

**Expected gain**: 14 × (3ms → <1ms) = 28ms saved/frame → ~20ms/frame total → **~50 fps** theoretical, probably **25-30 fps** realistic.

**Integration point**: In `talker.cpp` `predict_code_groups()`, add a third path:
```cpp
if (cp_cann_engine_ && cp_cann_engine_->is_ready()) {
    // Native CANN path — bypasses llama.cpp
    cp_cann_engine_->reset_kv_cache();
    cp_cann_engine_->forward_one_token(hidden_states, 0, cp_out.data());
    // ... same autoregressive loop as CPU path (lines 1528-1549) ...
}
```

### 2. xvec Audio Quality (MEDIUM — Issue #1)

xvec voice clone produces speech-range audio but sounds worse than MLX. Needs tensor-by-tensor debugging:
- Compare prefill hidden states layer-by-layer between C++ and MLX
- The `verify_*.py` scripts in `tools/qwen_tts/` can help
- Likely cause: weight precision (Q8_0 GGUF vs f32 MLX, especially codec_head/text_projection)

### 3. Push All Commits (LOW — needs user approval)

None of the repos have been pushed. All changes are local commits:
- OminiX-MLX: `git push` on main
- OminiX-API: `git push` on main
- OminiX-Ascend: `git push` on main

### 4. Open GitHub Issues (some done, some not)

**Already opened:**
- oxiglade/mlx-rs#339 — Flash attention for ASR/FLUX
- oxiglade/mlx-rs#340 — F16 KV cache
- oxiglade/mlx-rs#341 — Speculative TTS decode
- OminiX-ai/OminiX-Ascend#1 — xvec noise debugging
- OminiX-ai/OminiX-Ascend#2 — CV-specific GGUF export

**Should open:**
- OminiX-ai/OminiX-Ascend#3 — Native CANN CP engine (bypass llama.cpp)
- OminiX-ai/OminiX-Ascend#4 — CANN graph mode crash report (for Huawei)

## Key Files to Know

### OminiX-MLX
- `mlx-rs-core/src/memory.rs` — new memory module
- `mlx-rs-core/src/metal_kernels.rs` — flash attention kernel + fused ops
- `qwen3-tts-mlx/src/talker.rs` — batched QKV, generation
- `qwen3-tts-mlx/src/generate.rs` — MemoryGuard integration

### OminiX-Ascend
- `tools/qwen_tts/talker.cpp` — all TTS generation paths (ICL/xvec/customvoice)
- `tools/qwen_tts/talker.h` — TalkerSamplingParams (cp_max_groups, cp_max_layers)
- `tools/qwen_tts/qwen_tts.cpp` — orchestration, model loading
- `tools/qwen_tts/main.cpp` — CLI flags
- `tools/qwen_tts/cp_cann_engine.{h,cpp}` — native CANN prototype (NOT compiled yet)
- `tools/qwen_tts/export_qwen_tts.py` — GGUF export (now with spk_ids_json)
- `tools/qwen_tts/export_talker_llama.py` — llama-format GGUF export

### OminiX-API
- `src/engines/image.rs` — eval_with_retry, preflight, memory logging
- `src/engines/llm.rs` — memory logging
- `src/engines/qwen3_tts.rs` — memory logging

## Benchmarking Commands

### MLX TTS (Mac)
```bash
cd /Users/yuechen/home/OminiX-MLX
./target/release/examples/synthesize -m models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit -s vivian -l english -o /tmp/test.wav --seed 42 "Test text here."
```

### Ascend TTS (ICL, best quality)
```bash
./build/bin/qwen_tts -m tools/qwen_tts/gguf \
  -t "Test text." \
  -r tools/qwen_tts/data/ref_audios/ellen_ref_24k.wav \
  --ref_text "Reference transcript here." \
  --target_lang English --ref_lang English \
  --talker_model tools/qwen_tts/gguf/qwen_tts_talker_llama_q8_0.gguf \
  --cp_model tools/qwen_tts/gguf/qwen_tts_cp_llama.gguf \
  --n_gpu_layers 29 -n 8 -o output.wav -p
```

### Ascend TTS (CustomVoice)
```bash
./build/bin/qwen_tts -m tools/qwen_tts/gguf_cv \
  --mode customvoice --speaker vivian \
  -t "Test text." --target_lang English \
  --talker_model tools/qwen_tts/gguf_cv/qwen_tts_talker_llama_q8_0.gguf \
  --cp_model tools/qwen_tts/gguf_cv/qwen_tts_cp_llama.gguf \
  --n_gpu_layers 29 -n 8 -o output.wav
```

### Audio Quality Validation
```python
import wave, array
with wave.open('file.wav', 'rb') as w:
    frames = w.readframes(min(w.getnframes(), 24000*5))
s = array.array('h', frames)
rms = (sum(x*x for x in s) / len(s)) ** 0.5
zc = sum(1 for i in range(1,len(s)) if (s[i]>=0)!=(s[i-1]>=0))
zcr = zc / len(s) * 24000
print(f'RMS={rms:.0f} ZCR={zcr:.0f}Hz {"SPEECH" if 500<zcr<4000 and rms>1000 else "POOR"}')
# BUT: metrics can't distinguish intelligible speech from garbled speech-like audio.
# Always listen to the file. The user is strict about quality.
```

## Important Lessons Learned

1. **MLX lazy eval is very effective** — don't try to out-optimize it with custom kernels for small ops
2. **Q8_0 is fastest on Ascend** — Q5/Q4 are SLOWER due to unoptimized CANN kernels
3. **Standard RoPE works for TTS** — MRoPE GGUF breaks it (temporal-only positions = standard)
4. **All 5 CP layers needed** — can't reduce without quality collapse
5. **All 15 codec groups needed** — can't skip without EOS breaking + quality loss
6. **llama.cpp IS the bottleneck** on Ascend — 2ms per-launch overhead × 14 CP decodes = 28ms wasted
7. **User is strict about audio quality** — "sounds good" by metrics doesn't mean acceptable. Always play the file.
8. **File transfers to China servers**: Use AutoDL as relay (both in China). `/tmp` on AutoDL is FULL — use `/root/autodl-tmp/`.
