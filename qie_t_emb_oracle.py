#!/usr/bin/env python3
"""
Q2.4.5.5.20 - Pure-F32 numpy oracle for QIE-Edit time_text_embed chain.

After §5.5.19 GREEN-B (mod1 dispatch is bit-accurate when given native
t_emb as input), drift is upstream of mod1 - i.e. in t_emb itself or
the chain that produces it.

Engine path (image_diffusion_engine.cpp:5150-5223):
  1. sinusoidal[256]    = host_timestep_embedding_f32(t=sigma*1000, dim=256, max_period=10000)
                          layout: [cos(arg_0..arg_127), sin(arg_0..arg_127)]
  2. cast F32->F16
  3. t_emb_mid_f16[H]   = sinusoidal_f16 @ time_linear1.W^T + time_linear1.b   (dispatch_matmul_)
  4. silu in-place      (aclnnSilu)
  5. t_emb_out_f16[H]   = silu_t_f16 @ time_linear2.W^T + time_linear2.b       (dispatch_matmul_)
  6. dump 00_t_emb.f32  (F16 cast back to F32)

NOTE: QIE-Edit has NO text_embedder. time_text_embed = timestep_embedder ONLY.

Dump origin: /tmp/qie_dumps_5516/ from qie_q45_real_denoise_smoke with
make_flow_sigmas(20) -> sigmas[0]=1.0 -> t_val=1000.0 (FIRST step).
"""
import os
import sys
import numpy as np


DUMP = os.environ.get("QIE_DUMPS", "/tmp/qie_dumps_5516")
GGUF = os.environ.get(
    "QIE_GGUF",
    "/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf",
)
H = 3072


def find_gguf_py(repo_root):
    try:
        import gguf  # noqa: F401
        return
    except ImportError:
        pass
    cand = os.path.join(repo_root, "gguf-py")
    if os.path.isdir(cand):
        sys.path.insert(0, cand)


def load_dequant_weight(gguf_path, name):
    from gguf import GGUFReader, GGMLQuantizationType, quants
    reader = GGUFReader(gguf_path)
    tensor = None
    for t in reader.tensors:
        if t.name == name:
            tensor = t
            break
    if tensor is None:
        raise RuntimeError(f"tensor not found: {name}")
    K = int(tensor.shape[0])
    N = int(tensor.shape[1])
    qt = tensor.tensor_type
    raw = np.asarray(tensor.data, dtype=np.uint8).reshape(-1)

    if qt == GGMLQuantizationType.F32:
        return raw.view(np.float32).reshape(N, K).astype(np.float32), "F32"
    if qt == GGMLQuantizationType.F16:
        return raw.view(np.float16).reshape(N, K).astype(np.float32), "F16"
    if qt == GGMLQuantizationType.BF16:
        u16 = raw.view(np.uint16).astype(np.uint32)
        return ((u16 << 16).view(np.float32)).reshape(N, K).astype(np.float32), "BF16"

    qcls_map = {q.qtype: q for q in quants.__Quant.__subclasses__()
                if hasattr(q, "qtype")}
    qcls = qcls_map.get(qt)
    if qcls is None:
        raise RuntimeError(f"no Python dequant for type {qt}")
    from gguf.constants import GGML_QUANT_SIZES
    elems_per_blk, bytes_per_blk = GGML_QUANT_SIZES[qt]
    n_total = N * K
    n_blocks = n_total // elems_per_blk
    blocks = raw.reshape(n_blocks, bytes_per_blk)
    dq = qcls.dequantize_blocks(blocks)
    qt_name = str(qt).split(".")[-1]
    return dq.reshape(N, K).astype(np.float32), qt_name


def load_bias(gguf_path, name):
    from gguf import GGUFReader, GGMLQuantizationType
    reader = GGUFReader(gguf_path)
    for t in reader.tensors:
        if t.name == name:
            data = np.asarray(t.data)
            qt = t.tensor_type
            if qt == GGMLQuantizationType.F32:
                return data.astype(np.float32).reshape(-1).copy(), "F32"
            if qt == GGMLQuantizationType.F16:
                return data.astype(np.float32).reshape(-1), "F16"
            if qt == GGMLQuantizationType.BF16:
                u16 = data.view(np.uint16).astype(np.uint32)
                return ((u16 << 16).view(np.float32)).reshape(-1).astype(np.float32), "BF16"
            raise RuntimeError(f"bias dtype unsupported: {qt}")
    raise RuntimeError(f"bias not found: {name}")


def cossim(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def f16_round(x):
    return x.astype(np.float16).astype(np.float32)


def silu(x):
    return (x * (1.0 / (1.0 + np.exp(-x.astype(np.float64))))).astype(np.float32)


def stats(label, x):
    print(f"  {label:36s}  mean={x.mean():+.4e}  std={x.std():.4e}  "
          f"min={x.min():+.4e}  max={x.max():+.4e}  absmax={np.abs(x).max():.4e}")


def host_timestep_embedding_f32(t, dim=256, max_period=10000):
    """Engine layout (image_diffusion_engine.cpp:4626-4647):
      [cos(arg_0..arg_half-1), sin(arg_0..arg_half-1)]
      arg_j = t * exp(-log(max_period) * j / half)
    """
    out = np.zeros(dim, dtype=np.float32)
    half = dim // 2
    j = np.arange(half, dtype=np.float32)
    freq = np.exp(-np.log(float(max_period)) * j / float(half))
    arg = float(t) * freq
    out[:half] = np.cos(arg).astype(np.float32)
    out[half:2*half] = np.sin(arg).astype(np.float32)
    return out


def host_timestep_interleaved(t, dim=256, max_period=10000):
    """Alt layout: [sin0, cos0, sin1, cos1, ...]."""
    out = np.zeros(dim, dtype=np.float32)
    half = dim // 2
    j = np.arange(half, dtype=np.float32)
    freq = np.exp(-np.log(float(max_period)) * j / float(half))
    arg = float(t) * freq
    out[0::2] = np.sin(arg).astype(np.float32)
    out[1::2] = np.cos(arg).astype(np.float32)
    return out


def host_timestep_sin_cos(t, dim=256, max_period=10000):
    """Alt: [sin..., cos...] (HF default)."""
    out = np.zeros(dim, dtype=np.float32)
    half = dim // 2
    j = np.arange(half, dtype=np.float32)
    freq = np.exp(-np.log(float(max_period)) * j / float(half))
    arg = float(t) * freq
    out[:half] = np.sin(arg).astype(np.float32)
    out[half:2*half] = np.cos(arg).astype(np.float32)
    return out


def run_chain(sinu_f32, W1, b1, W2, b2, mode="f16"):
    """Replicate engine: F32 sinu -> cast F16 -> matmul1(F16 weights) -> silu -> matmul2(F16 weights)."""
    if mode == "f16":
        x = f16_round(sinu_f32)
        W1_ = f16_round(W1); b1_ = f16_round(b1)
        W2_ = f16_round(W2); b2_ = f16_round(b2)
        h = x @ W1_.T + b1_
        h = f16_round(h)              # post linear1 (engine stores F16)
        sh = f16_round(silu(h))       # post silu  (engine stores F16)
        y = sh @ W2_.T + b2_
        y = f16_round(y)              # post linear2 (engine stores F16)
        return x, h, sh, y
    else:  # pure F32 reference
        x = sinu_f32
        h = x @ W1.T + b1
        sh = silu(h)
        y = sh @ W2.T + b2
        return x, h, sh, y


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              "..", "..", ".."))
    find_gguf_py(repo_root)

    print(f"GGUF       : {GGUF}")
    print(f"DUMPS      : {DUMP}")
    print(f"H          : {H}")
    print()

    # ---- 1. native t_emb dump ----
    t_emb_native = np.fromfile(os.path.join(DUMP, "00_t_emb.f32"),
                                 dtype=np.float32)
    assert t_emb_native.size == H, f"shape mismatch: {t_emb_native.size}"
    print("--- native 00_t_emb.f32 stats ---")
    stats("00_t_emb (native)", t_emb_native)
    print()

    # ---- 2. weights ----
    print("--- loading time_text_embed weights ---")
    W1, W1_qt = load_dequant_weight(GGUF, "time_text_embed.timestep_embedder.linear_1.weight")
    b1, b1_qt = load_bias(GGUF, "time_text_embed.timestep_embedder.linear_1.bias")
    W2, W2_qt = load_dequant_weight(GGUF, "time_text_embed.timestep_embedder.linear_2.weight")
    b2, b2_qt = load_bias(GGUF, "time_text_embed.timestep_embedder.linear_2.bias")
    print(f"  linear_1.weight  shape={W1.shape}  qt={W1_qt}")
    print(f"  linear_1.bias    shape={b1.shape}   qt={b1_qt}")
    print(f"  linear_2.weight  shape={W2.shape}  qt={W2_qt}")
    print(f"  linear_2.bias    shape={b2.shape}   qt={b2_qt}")
    if W1.shape != (H, 256):
        print(f"  WARN: W1 shape != (H={H}, 256) — checking transpose...")
    if W2.shape != (H, H):
        print(f"  WARN: W2 shape != (H, H)")
    stats("W1", W1); stats("b1", b1); stats("W2", W2); stats("b2", b2)
    print()

    # ---- 3. timestep ----
    # Run came from qie_q45_real_denoise_smoke -> make_flow_sigmas(20) ->
    # sigmas[0]=1.0 -> t_val = 1000.0
    t_val_default = 1000.0
    print(f"--- primary candidate t_val={t_val_default} (sigma=1.0 * 1000) ---")

    # ---- 4. main oracle (engine layout: cos-then-sin) ----
    sinu = host_timestep_embedding_f32(t_val_default, 256, 10000)
    stats("sinu_engine [cos|sin]", sinu)
    x, h, sh, y = run_chain(sinu, W1, b1, W2, b2, mode="f16")
    stats("post_linear1 (F16-RT)", h)
    stats("post_silu    (F16-RT)", sh)
    stats("post_linear2 = oracle t_emb (F16-RT)", y)
    cos_full = cossim(y, t_emb_native)
    print(f"\n  ORACLE COS (engine [cos|sin], t=1000, F16-RT): {cos_full:.6f}")
    print(f"  max_abs_diff = {np.abs(y - t_emb_native).max():.4e}")
    print(f"  mean_abs_diff = {np.abs(y - t_emb_native).mean():.4e}")
    print()

    # ---- 5. alt-config sweep ----
    print("=" * 70)
    print("ALT-CONFIG SWEEP")
    print("=" * 70)

    configs = [
        ("engine [cos|sin] t=1000 F16",   sinu, "f16"),
        ("engine [cos|sin] t=1000 F32",   sinu, "f32"),
        ("HF [sin|cos]     t=1000 F16",   host_timestep_sin_cos(1000.0), "f16"),
        ("interleaved      t=1000 F16",   host_timestep_interleaved(1000.0), "f16"),
        ("engine [cos|sin] t=999  F16",   host_timestep_embedding_f32(999.0), "f16"),
        ("engine [cos|sin] t=1    F16",   host_timestep_embedding_f32(1.0),   "f16"),
        ("engine [cos|sin] t=950  F16",   host_timestep_embedding_f32(950.0), "f16"),
        ("engine [cos|sin] t=500  F16",   host_timestep_embedding_f32(500.0), "f16"),
    ]
    for label, s, mode in configs:
        _, _, _, yy = run_chain(s, W1, b1, W2, b2, mode=mode)
        c = cossim(yy, t_emb_native)
        amx = np.abs(yy).max()
        print(f"  {label:38s}  cos={c:+.6f}  oracle_absmax={amx:.4e}")
    print()

    # ---- 6. test transposed weights (W2[K,N] vs [N,K]) ----
    print("--- weight-orientation sanity check ---")
    # If load returned (N,K) but engine expects (K,N), .T is wrong direction.
    try:
        sinu_eng = host_timestep_embedding_f32(1000.0)
        x_ = f16_round(sinu_eng)
        W1_alt = f16_round(W1)  # try without transpose: x @ W1 (no .T)
        if W1_alt.shape[0] == 256:
            h_alt = x_ @ W1_alt + f16_round(b1)
            print(f"  W1 no-transpose path: shape would be {h_alt.shape}")
    except Exception as e:
        print(f"  no-transpose check err: {e}")
    print()

    # ---- 7. magnitude vs §5.5.7 historical ----
    print("--- magnitude reality check ---")
    print(f"  Current native 00_t_emb absmax: {np.abs(t_emb_native).max():.4e}")
    print(f"  §5.5.7 historical               : 1.118e+02")
    print(f"  Oracle absmax (F16, t=1000)     : {np.abs(y).max():.4e}")
    print(f"  Engine path is QUANTIZED weights (Q4_0/Q5_K) -> dequant may shift")
    print()

    # ---- decision ----
    print("=" * 70)
    if cos_full >= 0.999:
        verdict = "GREEN-B (chain bit-accurate)"
    elif cos_full >= 0.99:
        verdict = "AMBER (near-bit-accurate, F16 drift only)"
    else:
        verdict = "GREEN-A (CHAIN BUG)"
    print(f"VERDICT: {verdict}  cos={cos_full:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
