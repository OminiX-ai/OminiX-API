#!/usr/bin/env python3
"""
Qwen-Image-Edit-2511 MLX inference (P0 optimized).

P0 changes vs infer_qwen_image_edit.py:
  - residual_dtype + attention_dtype: float32 -> bfloat16
  - CFG batched: cond+uncond run as batch=2 in a single forward, halving
    weight-bandwidth pressure on the Q4_K_M projections.
"""
from __future__ import annotations

import argparse
import gc
import math
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from infer_qwen_gguf import (
    GGUFQwen,
    build_hf_tokenizer,
    build_tokenizer,
    normalize_token_ids,
    resolve_affine_quant_params,
)


PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
    "{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_DROP_IDX = 34

# Edit-2511 vision prompt template (matches diffusers' QwenImageEditPlusPipeline).
# The `{}` slots for user text go at the end; image blocks are inserted before it.
EDIT_PROMPT_TEMPLATE_SYSTEM = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. "
    "Generate a new image that meets the user's requirements while maintaining consistency with "
    "the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n"
)
EDIT_PROMPT_TEMPLATE_ASSISTANT = "<|im_end|>\n<|im_start|>assistant\n"
EDIT_PROMPT_TEMPLATE_DROP_IDX = 64  # tokens before user content

# Default HF tokenizer repo used when the user supplies a vision reference image.
QWEN25_VL_HF_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"

# CLIP-style normalization used by Qwen2.5-VL image processor.
QWEN_VL_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
QWEN_VL_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

# Vision token id for Qwen/Qwen2.5-VL-7B-Instruct tokenizer. Only the pad id is
# needed for runtime scatter; start/end are emitted by the tokenizer from template
# text and don't require numeric lookups.
QWEN_VL_IMAGE_PAD_ID = 151655
QWEN_IMAGE_LATENTS_MEAN = [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
]
QWEN_IMAGE_LATENTS_STD = [
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
]


LinearFn = Callable[[mx.array], mx.array]


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _gelu_approx(x: mx.array) -> mx.array:
    return nn.gelu_approx(x)


def _layer_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    x32 = x.astype(mx.float32)
    mean = mx.mean(x32, axis=-1, keepdims=True)
    var = mx.mean((x32 - mean) * (x32 - mean), axis=-1, keepdims=True)
    y = (x32 - mean) * mx.rsqrt(var + eps)
    return y.astype(x.dtype)


def _rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    x32 = x.astype(mx.float32)
    rrms = mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    y = (x32 * rrms).astype(x.dtype)
    return y * weight.astype(y.dtype)


def _rms_norm_channel_first(
    x: mx.array,
    gamma: mx.array,
    *,
    eps: float = 1e-12,
) -> mx.array:
    x32 = x.astype(mx.float32)
    channels = int(x.shape[1])
    norm = mx.sqrt(mx.maximum(mx.sum(x32 * x32, axis=1, keepdims=True), eps))
    y = x32 * (math.sqrt(channels) / norm)
    return y.astype(x.dtype) * gamma.astype(y.dtype)


def _reshape_for_heads(
    x: mx.array,
    *,
    num_heads: int,
    head_dim: int,
) -> mx.array:
    batch, seq_len, _ = x.shape
    return x.reshape((batch, seq_len, num_heads, head_dim)).transpose((0, 2, 1, 3))


def _merge_heads(x: mx.array) -> mx.array:
    batch, heads, seq_len, head_dim = x.shape
    return x.transpose((0, 2, 1, 3)).reshape((batch, seq_len, heads * head_dim))


def _clip_fp16(x: mx.array) -> mx.array:
    if x.dtype == mx.float16:
        return mx.clip(x, -65504.0, 65504.0)
    return x


def _timestep_embedding(
    timesteps: mx.array,
    dim: int,
    *,
    max_period: float = 10000.0,
    time_factor: float = 1000.0,
) -> mx.array:
    timesteps = timesteps.astype(mx.float32) * time_factor
    half = dim // 2
    freqs = mx.exp(
        -math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / max(half, 1)
    )
    args = timesteps[:, None] * freqs[None, :]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1)
    return emb


def _norm_last_dim(x: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.sqrt(mx.maximum(mx.sum(x.astype(mx.float32) ** 2, axis=-1, keepdims=True), eps))


def _conv2d_weight(weight: mx.array) -> mx.array:
    return weight.transpose((0, 2, 3, 1))


def _conv3d_weight(weight: mx.array) -> mx.array:
    return weight.transpose((0, 2, 3, 4, 1))


def _conv2d_nchw(
    x: mx.array,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
) -> mx.array:
    x_hwc = x.transpose((0, 2, 3, 1))
    y = mx.conv2d(
        x_hwc,
        _conv2d_weight(weight),
        stride=stride,
        padding=padding,
    )
    if bias is not None:
        y = y + bias.astype(y.dtype)
    return y.transpose((0, 3, 1, 2))


def _causal_conv3d_ncthw(
    x: mx.array,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0),
) -> mx.array:
    pad_t, pad_h, pad_w = padding
    if pad_t or pad_h or pad_w:
        x = mx.pad(
            x,
            [
                (0, 0),
                (0, 0),
                (2 * pad_t, 0),
                (pad_h, pad_h),
                (pad_w, pad_w),
            ],
        )
    x_dhwc = x.transpose((0, 2, 3, 4, 1))
    y = mx.conv3d(
        x_dhwc,
        _conv3d_weight(weight),
        stride=stride,
        padding=(0, 0, 0),
    )
    if bias is not None:
        y = y + bias.astype(y.dtype)
    return y.transpose((0, 4, 1, 2, 3))


def _upsample_spatial_2x_ncthw(x: mx.array) -> mx.array:
    batch, channels, time, height, width = x.shape
    x = x.reshape((batch, channels, time, height, 1, width, 1))
    x = mx.broadcast_to(x, (batch, channels, time, height, 2, width, 2))
    return x.reshape((batch, channels, time, height * 2, width * 2))


def _flatten_video_frames(x: mx.array) -> tuple[mx.array, int, int]:
    batch, channels, time, height, width = x.shape
    return x.transpose((0, 2, 1, 3, 4)).reshape((batch * time, channels, height, width)), batch, time


def _restore_video_frames(x: mx.array, batch: int, time: int) -> mx.array:
    _, channels, height, width = x.shape
    return x.reshape((batch, time, channels, height, width)).transpose((0, 2, 1, 3, 4))


def _pack_latents(latents: mx.array) -> mx.array:
    batch, channels, _, height, width = latents.shape
    latents = latents.reshape((batch, channels, height // 2, 2, width // 2, 2))
    latents = latents.transpose((0, 2, 4, 1, 3, 5))
    return latents.reshape((batch, (height // 2) * (width // 2), channels * 4))


def _unpack_latents(latents: mx.array, height: int, width: int, vae_scale_factor: int = 8) -> mx.array:
    batch, _, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.reshape((batch, height // 2, width // 2, channels // 4, 2, 2))
    latents = latents.transpose((0, 3, 1, 4, 2, 5))
    return latents.reshape((batch, channels // 4, 1, height, width))


def _image_to_uint8(x: mx.array) -> np.ndarray:
    x = x.astype(mx.float32)
    x = mx.clip((x + 1.0) * 127.5, 0.0, 255.0)
    x = x.transpose((1, 2, 0))
    arr = np.array(x)
    if not np.isfinite(arr).all():
        raise ValueError(
            "Decoded image contains NaN/Inf values. This indicates numerical divergence "
            "during denoising or VAE decode, so the image is not valid."
        )
    return arr.astype(np.uint8)


def _infer_compute_dtype(weights: dict[str, mx.array]) -> mx.Dtype:
    dtypes = {
        arr.dtype
        for arr in weights.values()
        if hasattr(arr, "dtype") and mx.issubdtype(arr.dtype, mx.floating)
    }
    for dtype in (mx.bfloat16, mx.float16, mx.float32):
        if dtype in dtypes:
            return dtype
    return mx.float16


def _print_stats(name: str, x: mx.array, *, enabled: bool) -> None:
    if not enabled:
        return
    arr = np.array(x.astype(mx.float32))
    finite = np.isfinite(arr)
    finite_count = int(finite.sum())
    total = int(arr.size)
    if finite_count == 0:
        print(f"[stats] {name}: shape={arr.shape} finite=0/{total}")
        return
    valid = arr[finite]
    print(
        f"[stats] {name}: shape={arr.shape} finite={finite_count}/{total} "
        f"min={valid.min():.6f} max={valid.max():.6f} mean={valid.mean():.6f}"
    )


def _load_gguf(path: str) -> tuple[dict[str, mx.array], dict[str, Any]]:
    loaded = mx.load(str(Path(path).expanduser()), return_metadata=True)
    if isinstance(loaded, tuple):
        return loaded
    return loaded, {}


def _load_safetensors(path: str) -> dict[str, mx.array]:
    loaded = mx.load(str(Path(path).expanduser()))
    if isinstance(loaded, tuple):
        return loaded[0]
    return loaded


def _guess_tokenizer_ref(llm_path: Path) -> str | None:
    stem = llm_path.stem
    prefix = stem.split("-Q", 1)[0]
    candidates = [
        llm_path.parent / prefix,
        llm_path.parent / stem,
    ]
    for candidate in candidates:
        if (candidate / "tokenizer.json").exists() or (candidate / "config.json").exists():
            return str(candidate)
    return None


def _pick_tokenizer(
    meta: dict[str, Any],
    *,
    tokenizer_ref: str | None,
    trust_remote_code: bool,
) -> tuple[Any, str]:
    if tokenizer_ref is not None:
        return build_hf_tokenizer(tokenizer_ref, trust_remote_code=trust_remote_code), "hf"
    tokenizer = build_tokenizer(meta)
    if tokenizer is None:
        raise ValueError(
            "Failed to construct a tokenizer from GGUF metadata. Pass --tokenizer with a HF tokenizer path/repo."
        )
    return tokenizer, "gguf"


def _tokenize_prompt(
    tokenizer: Any,
    prompt_text: str,
    *,
    max_length: int,
    source: str,
) -> list[int]:
    if source == "hf":
        encoded = tokenizer(
            prompt_text,
            max_length=max_length,
            truncation=True,
            return_attention_mask=False,
        )
        return normalize_token_ids(encoded)
    token_ids = list(tokenizer.encode(prompt_text))
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    return [int(x) for x in token_ids]


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _sigma_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    base = np.linspace(1.0, 1.0 / num_steps, num_steps, dtype=np.float32)
    mu = _calculate_shift(image_seq_len)
    shifted = math.exp(mu) / (math.exp(mu) + np.power((1.0 / base) - 1.0, 1.0))
    sigmas = shifted.astype(np.float32).tolist()
    sigmas.append(0.0)
    return sigmas


def _sigma_schedule_flow_shift(num_steps: int, flow_shift: float) -> list[float]:
    # Matches sd.cpp DiscreteFlowDenoiser + DiscreteScheduler:
    # t ∈ [999, 0] with (n-1) steps; t_normalized = (t+1)/1000 ∈ [1.0, 0.001].
    timesteps = 1000
    t_max = timesteps - 1
    if num_steps <= 1:
        base = np.array([1.0], dtype=np.float32)
    else:
        step = t_max / (num_steps - 1)
        t_arr = t_max - step * np.arange(num_steps, dtype=np.float32)
        base = (t_arr + 1.0) / float(timesteps)
    if flow_shift == 1.0:
        shifted = base
    else:
        shifted = flow_shift * base / (1.0 + (flow_shift - 1.0) * base)
    sigmas = shifted.astype(np.float32).tolist()
    sigmas.append(0.0)
    return sigmas


def _sample_latent_noise(
    *,
    seed: int,
    height: int,
    width: int,
    channels: int,
    dtype: mx.Dtype,
) -> mx.array:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((1, channels, 1, height, width), dtype=np.float32)
    return mx.array(noise).astype(dtype)


def _rope_frequencies(positions: mx.array, dim: int, theta: float = 10000.0) -> tuple[mx.array, mx.array]:
    if dim % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {dim}.")
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    omega = 1.0 / (theta**scale)
    angles = positions.astype(mx.float32)[:, None] * omega[None, :]
    return mx.cos(angles), mx.sin(angles)


def _centered_positions(length: int) -> mx.array:
    left = mx.arange(-(length - length // 2), 0, dtype=mx.float32)
    right = mx.arange(0, length // 2, dtype=mx.float32)
    return mx.concatenate([left, right], axis=0)


def _build_qwen_image_rope(
    *,
    img_shape: tuple[int, int, int],
    txt_seq_len: int,
    theta: float = 10000.0,
    axes_dims: tuple[int, int, int] = (16, 56, 56),
) -> tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]]:
    frame, height, width = img_shape
    frame_pos = mx.arange(frame, dtype=mx.float32)
    height_pos = _centered_positions(height)
    width_pos = _centered_positions(width)

    frame_cos, frame_sin = _rope_frequencies(frame_pos, axes_dims[0], theta)
    height_cos, height_sin = _rope_frequencies(height_pos, axes_dims[1], theta)
    width_cos, width_sin = _rope_frequencies(width_pos, axes_dims[2], theta)

    frame_cos = mx.broadcast_to(frame_cos[:, None, None, :], (frame, height, width, frame_cos.shape[-1]))
    frame_sin = mx.broadcast_to(frame_sin[:, None, None, :], (frame, height, width, frame_sin.shape[-1]))
    height_cos = mx.broadcast_to(height_cos[None, :, None, :], (frame, height, width, height_cos.shape[-1]))
    height_sin = mx.broadcast_to(height_sin[None, :, None, :], (frame, height, width, height_sin.shape[-1]))
    width_cos = mx.broadcast_to(width_cos[None, None, :, :], (frame, height, width, width_cos.shape[-1]))
    width_sin = mx.broadcast_to(width_sin[None, None, :, :], (frame, height, width, width_sin.shape[-1]))

    img_cos = mx.concatenate([frame_cos, height_cos, width_cos], axis=-1).reshape((frame * height * width, -1))
    img_sin = mx.concatenate([frame_sin, height_sin, width_sin], axis=-1).reshape((frame * height * width, -1))

    # scale_rope=True: diffusers/mflux uses halved positions for txt_offset.
    txt_offset = max(height // 2, width // 2)
    txt_pos = mx.arange(txt_offset, txt_offset + txt_seq_len, dtype=mx.float32)
    txt_cos, txt_sin = [], []
    for dim in axes_dims:
        cos, sin = _rope_frequencies(txt_pos, dim, theta)
        txt_cos.append(cos)
        txt_sin.append(sin)
    return (img_cos, img_sin), (mx.concatenate(txt_cos, axis=-1), mx.concatenate(txt_sin, axis=-1))


def _gen_img_position_grid(
    shape: tuple[int, int, int],
    *,
    index: int,
    centered: bool,
    h_offset: int = 0,
    w_offset: int = 0,
) -> tuple[mx.array, mx.array, mx.array]:
    frame, height, width = shape
    h_start = h_offset - (height // 2 if centered else 0)
    w_start = w_offset - (width // 2 if centered else 0)
    row_pos = mx.arange(h_start, h_start + height, dtype=mx.float32)
    col_pos = mx.arange(w_start, w_start + width, dtype=mx.float32)
    frame_grid = mx.full((frame, height, width), float(index), dtype=mx.float32).reshape(-1)
    row_grid = mx.broadcast_to(row_pos[None, :, None], (frame, height, width)).reshape(-1)
    col_grid = mx.broadcast_to(col_pos[None, None, :], (frame, height, width)).reshape(-1)
    return frame_grid, row_grid, col_grid


def _build_qwen_image_edit_rope(
    *,
    img_shape: tuple[int, int, int],
    ref_shapes: list[tuple[int, int, int]],
    txt_seq_len: int,
    theta: float = 10000.0,
    axes_dims: tuple[int, int, int] = (16, 56, 56),
    increase_ref_index: bool = False,
) -> tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]]:
    # Matches diffusers QwenEmbedRope with scale_rope=True: every image
    # (main and each ref) uses CENTERED spatial positions ([-h//2, h//2-1]).
    # Images are distinguished only by the frame axis index (main=0, ref=1, ...).
    all_shapes = [img_shape, *ref_shapes]
    frame_parts: list[mx.array] = []
    row_parts: list[mx.array] = []
    col_parts: list[mx.array] = []

    max_vid_index = 0
    for idx, shape in enumerate(all_shapes):
        frame, height, width = shape
        frame_pos = mx.arange(idx, idx + frame, dtype=mx.float32)
        height_pos = _centered_positions(height)
        width_pos = _centered_positions(width)
        frame_grid = mx.broadcast_to(frame_pos[:, None, None], (frame, height, width)).reshape(-1)
        row_grid = mx.broadcast_to(height_pos[None, :, None], (frame, height, width)).reshape(-1)
        col_grid = mx.broadcast_to(width_pos[None, None, :], (frame, height, width)).reshape(-1)
        frame_parts.append(frame_grid)
        row_parts.append(row_grid)
        col_parts.append(col_grid)
        max_vid_index = max(max_vid_index, height // 2, width // 2)

    frame_all = mx.concatenate(frame_parts, axis=0)
    row_all = mx.concatenate(row_parts, axis=0)
    col_all = mx.concatenate(col_parts, axis=0)

    frame_cos, frame_sin = _rope_frequencies(frame_all, axes_dims[0], theta)
    height_cos, height_sin = _rope_frequencies(row_all, axes_dims[1], theta)
    width_cos, width_sin = _rope_frequencies(col_all, axes_dims[2], theta)
    img_cos = mx.concatenate([frame_cos, height_cos, width_cos], axis=-1)
    img_sin = mx.concatenate([frame_sin, height_sin, width_sin], axis=-1)

    txt_pos = mx.arange(max_vid_index, max_vid_index + txt_seq_len, dtype=mx.float32)
    txt_cos, txt_sin = [], []
    for dim in axes_dims:
        cos, sin = _rope_frequencies(txt_pos, dim, theta)
        txt_cos.append(cos)
        txt_sin.append(sin)
    return (img_cos, img_sin), (mx.concatenate(txt_cos, axis=-1), mx.concatenate(txt_sin, axis=-1))


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    x_ = x.astype(mx.float32).reshape((*x.shape[:-1], -1, 2))
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    real = x_[..., 0]
    imag = x_[..., 1]
    out = mx.stack([real * cos - imag * sin, real * sin + imag * cos], axis=-1)
    return out.reshape(x.shape).astype(x.dtype)


class BiasAwareGGUFQwen(GGUFQwen):
    def _build_linear_fn(self, prefix: str) -> LinearFn:
        bias = self.w.get(f"{prefix}.bias")
        if self._has_quantized_triplet(prefix):
            params = self._get_quant_params(prefix)
            w = self.w[f"{prefix}.weight"]
            s = self.w[f"{prefix}.scales"]
            b = self.w[f"{prefix}.biases"]

            def linear(x: mx.array, w=w, s=s, b=b, bias=bias, params=params) -> mx.array:
                y = mx.quantized_matmul(
                    x,
                    w,
                    s,
                    b,
                    transpose=True,
                    group_size=params.group_size,
                    bits=params.bits,
                    mode=params.mode,
                )
                if bias is not None:
                    y = y + bias.astype(y.dtype)
                return y

            return linear

        w = self.w[f"{prefix}.weight"]

        def linear(x: mx.array, w=w, bias=bias) -> mx.array:
            y = x @ w.T
            if bias is not None:
                y = y + bias.astype(y.dtype)
            return y

        return linear


class Qwen25VLTextEncoder(BiasAwareGGUFQwen):
    def __init__(self, weights: dict[str, mx.array], meta: dict[str, Any]):
        super().__init__(weights, meta)
        if self.cfg.arch != "qwen2vl":
            raise ValueError(f"Expected qwen2vl GGUF, got {self.cfg.arch!r}.")

    def embed_tokens(self, token_ids: mx.array) -> mx.array:
        return self._embed(token_ids)

    def encode_last_hidden_state(
        self,
        token_ids: mx.array | None = None,
        *,
        inputs_embeds: mx.array | None = None,
    ) -> mx.array:
        if inputs_embeds is None:
            if token_ids is None:
                raise ValueError("encode_last_hidden_state requires either token_ids or inputs_embeds.")
            x = self._embed(token_ids).astype(self._compute_dtype)
        else:
            x = inputs_embeds.astype(self._compute_dtype)
        batch, seq_len, _ = x.shape

        for blk in self._blocks:
            resid = x
            x_norm = self._rms_norm(x, blk.attn_norm)
            q = blk.attn_q(x_norm)
            k = blk.attn_k(x_norm)
            v = blk.attn_v(x_norm)

            if blk.q_norm is not None:
                q = self._rms_norm(
                    q.reshape((batch, seq_len, self.cfg.n_heads, self.cfg.head_dim)),
                    blk.q_norm,
                )
            else:
                q = q.reshape((batch, seq_len, self.cfg.n_heads, self.cfg.head_dim))

            if blk.k_norm is not None:
                k = self._rms_norm(
                    k.reshape((batch, seq_len, self.cfg.n_kv_heads, self.cfg.head_dim)),
                    blk.k_norm,
                )
            else:
                k = k.reshape((batch, seq_len, self.cfg.n_kv_heads, self.cfg.head_dim))

            v = v.reshape((batch, seq_len, self.cfg.n_kv_heads, self.cfg.head_dim))

            q = q.transpose((0, 2, 1, 3))
            k = k.transpose((0, 2, 1, 3))
            v = v.transpose((0, 2, 1, 3))

            q = self._apply_rope(q, 0)
            k = self._apply_rope(k, 0)

            ctx = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=self._attn_scale,
                mask="causal",
            )
            ctx = _merge_heads(ctx)
            x = resid + blk.attn_output(ctx)

            resid = x
            x_norm = self._rms_norm(x, blk.ffn_norm)
            up = blk.ffn_up(x_norm)
            gate = blk.ffn_gate(x_norm)
            x = resid + blk.ffn_down(_silu(gate) * up)

        return self._rms_norm(x, self._output_norm_weight)


class QwenImageTransformer:
    def __init__(self, weights: dict[str, mx.array], meta: dict[str, Any]):
        del meta
        self.w = weights
        self.compute_dtype = _infer_compute_dtype(weights)
        self.mlp_lowp_dtype = mx.bfloat16
        self.residual_dtype = mx.bfloat16
        self.attention_dtype = mx.bfloat16
        self.in_channels = 64
        self.out_channels = 16
        self.patch_size = 2
        self.inner_dim = 3072
        self.joint_attention_dim = 3584
        self.num_heads = 24
        self.head_dim = 128
        self.num_layers = 60
        self.attn_scale = 1.0 / math.sqrt(self.head_dim)
        self.q_eps = 1e-6
        self._quant_params_cache: dict[str, Any] = {}

    def _weight(self, key: str) -> mx.array:
        if key not in self.w:
            raise KeyError(f"Missing weight: {key}")
        return self.w[key]

    def _has_quantized_triplet(self, prefix: str) -> bool:
        return (
            f"{prefix}.weight" in self.w
            and f"{prefix}.scales" in self.w
            and f"{prefix}.biases" in self.w
        )

    def _linear_prefix(self, prefix: str, x: mx.array) -> mx.array:
        bias = self.w.get(f"{prefix}.bias")
        if self._has_quantized_triplet(prefix):
            if prefix not in self._quant_params_cache:
                self._quant_params_cache[prefix] = resolve_affine_quant_params(self.w, {}, prefix)
            params = self._quant_params_cache[prefix]
            y = mx.quantized_matmul(
                x,
                self._weight(f"{prefix}.weight"),
                self._weight(f"{prefix}.scales"),
                self._weight(f"{prefix}.biases"),
                transpose=True,
                group_size=params.group_size,
                bits=params.bits,
                mode=params.mode,
            )
        else:
            y = x @ self._weight(f"{prefix}.weight").T
        if bias is not None:
            y = y + bias.astype(y.dtype)
        return y

    def _linear_lowp(
        self,
        prefix: str,
        x: mx.array,
        *,
        out_dtype: mx.Dtype | None = None,
    ) -> mx.array:
        y = self._linear_prefix(prefix, x.astype(self.mlp_lowp_dtype))
        if out_dtype is not None and y.dtype != out_dtype:
            y = y.astype(out_dtype)
        return y

    def _linear_full(
        self,
        prefix: str,
        x: mx.array,
        *,
        out_dtype: mx.Dtype | None = None,
    ) -> mx.array:
        y = self._linear_prefix(prefix, x.astype(self.residual_dtype))
        if out_dtype is not None and y.dtype != out_dtype:
            y = y.astype(out_dtype)
        return y

    def _time_embedding(self, timesteps: mx.array) -> mx.array:
        x = _timestep_embedding(timesteps, 256).astype(self.residual_dtype)
        x = self._linear_full(
            "time_text_embed.timestep_embedder.linear_1",
            x,
            out_dtype=self.residual_dtype,
        )
        x = _silu(x)
        return self._linear_full(
            "time_text_embed.timestep_embedder.linear_2",
            x,
            out_dtype=self.residual_dtype,
        )

    def _modulate(
        self,
        x: mx.array,
        shift: mx.array,
        scale: mx.array,
    ) -> mx.array:
        return x * (1.0 + scale[:, None, :]) + shift[:, None, :]

    def _modulate_flex(
        self,
        x: mx.array,
        shift: mx.array,
        scale: mx.array,
    ) -> mx.array:
        if shift.ndim == 2:
            return x * (1.0 + scale[:, None, :]) + shift[:, None, :]
        return x * (1.0 + scale) + shift

    def _gate_flex(self, gate: mx.array, y: mx.array) -> mx.array:
        if gate.ndim == 2:
            return gate[:, None, :] * y
        return gate * y

    def _prepare_img_mod(
        self,
        prefix: str,
        temb: mx.array,
        modulate_index: mx.array | None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        img_mod = self._linear_full(
            f"{prefix}.img_mod.1",
            _silu(temb),
            out_dtype=self.residual_dtype,
        )
        if modulate_index is None:
            return tuple(mx.split(img_mod, 6, axis=-1))  # each [N, hidden]
        parts = mx.split(img_mod, 6, axis=-1)  # 6 tensors of [2N, hidden] where row0=real, row1=zero
        idx = modulate_index.astype(self.residual_dtype)[None, :, None]  # [1, seq, 1]
        blended = []
        for p in parts:
            real = p[0:1]  # [1, hidden]
            zero = p[1:2]  # [1, hidden]
            blended.append((1.0 - idx) * real[:, None, :] + idx * zero[:, None, :])  # [1, seq, hidden]
        return tuple(blended)

    def _block(
        self,
        idx: int,
        img: mx.array,
        txt: mx.array,
        temb: mx.array,
        img_rope: tuple[mx.array, mx.array],
        txt_rope: tuple[mx.array, mx.array],
        modulate_index: mx.array | None = None,
        attn_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        prefix = f"transformer_blocks.{idx}"

        img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2 = self._prepare_img_mod(
            prefix, temb, modulate_index
        )
        real_temb = temb if modulate_index is None else temb[: temb.shape[0] // 2]
        txt_mod = self._linear_full(
            f"{prefix}.txt_mod.1",
            _silu(real_temb),
            out_dtype=self.residual_dtype,
        )
        txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = mx.split(txt_mod, 6, axis=-1)

        img_norm = _layer_norm(img, eps=self.q_eps)
        txt_norm = _layer_norm(txt, eps=self.q_eps)
        img_modulated = self._modulate_flex(img_norm, img_shift1, img_scale1)
        txt_modulated = self._modulate(txt_norm, txt_shift1, txt_scale1)

        img_q = _reshape_for_heads(
            self._linear_full(
                f"{prefix}.attn.to_q",
                img_modulated,
                out_dtype=self.attention_dtype,
            ),
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        img_k = _reshape_for_heads(
            self._linear_full(
                f"{prefix}.attn.to_k",
                img_modulated,
                out_dtype=self.attention_dtype,
            ),
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        img_v = _reshape_for_heads(
            self._linear_full(
                f"{prefix}.attn.to_v",
                img_modulated,
                out_dtype=self.attention_dtype,
            ),
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        txt_q = _reshape_for_heads(
            self._linear_full(
                f"{prefix}.attn.add_q_proj",
                txt_modulated,
                out_dtype=self.attention_dtype,
            ),
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        txt_k = _reshape_for_heads(
            self._linear_full(
                f"{prefix}.attn.add_k_proj",
                txt_modulated,
                out_dtype=self.attention_dtype,
            ),
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        txt_v = _reshape_for_heads(
            self._linear_full(
                f"{prefix}.attn.add_v_proj",
                txt_modulated,
                out_dtype=self.attention_dtype,
            ),
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        img_q = _rms_norm(img_q, self._weight(f"{prefix}.attn.norm_q.weight"), eps=self.q_eps)
        img_k = _rms_norm(img_k, self._weight(f"{prefix}.attn.norm_k.weight"), eps=self.q_eps)
        txt_q = _rms_norm(txt_q, self._weight(f"{prefix}.attn.norm_added_q.weight"), eps=self.q_eps)
        txt_k = _rms_norm(txt_k, self._weight(f"{prefix}.attn.norm_added_k.weight"), eps=self.q_eps)

        img_q = _apply_rope(img_q, img_rope[0], img_rope[1])
        img_k = _apply_rope(img_k, img_rope[0], img_rope[1])
        txt_q = _apply_rope(txt_q, txt_rope[0], txt_rope[1])
        txt_k = _apply_rope(txt_k, txt_rope[0], txt_rope[1])

        q = mx.concatenate([txt_q, img_q], axis=2).astype(self.attention_dtype)
        k = mx.concatenate([txt_k, img_k], axis=2).astype(self.attention_dtype)
        v = mx.concatenate([txt_v, img_v], axis=2).astype(self.attention_dtype)
        attn = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.attn_scale, mask=attn_mask
        )
        attn = _merge_heads(attn).astype(self.residual_dtype)

        txt_len = int(txt.shape[1])
        txt_attn = attn[:, :txt_len, :]
        img_attn = attn[:, txt_len:, :]

        img = img + self._gate_flex(
            img_gate1,
            self._linear_full(
                f"{prefix}.attn.to_out.0",
                img_attn,
                out_dtype=self.residual_dtype,
            ),
        )
        txt = txt + txt_gate1[:, None, :] * self._linear_full(
            f"{prefix}.attn.to_add_out",
            txt_attn,
            out_dtype=self.residual_dtype,
        )

        img_mlp_in = self._modulate_flex(_layer_norm(img, eps=self.q_eps), img_shift2, img_scale2)
        img_mlp_hidden = self._linear_lowp(
            f"{prefix}.img_mlp.net.0.proj",
            img_mlp_in,
            out_dtype=self.mlp_lowp_dtype,
        )
        img_mlp = self._linear_lowp(
            f"{prefix}.img_mlp.net.2",
            _gelu_approx(img_mlp_hidden),
            out_dtype=self.residual_dtype,
        )
        img = img + self._gate_flex(img_gate2, img_mlp)

        txt_mlp_in = self._modulate(_layer_norm(txt, eps=self.q_eps), txt_shift2, txt_scale2)
        txt_mlp_hidden = self._linear_lowp(
            f"{prefix}.txt_mlp.net.0.proj",
            txt_mlp_in,
            out_dtype=self.mlp_lowp_dtype,
        )
        txt_mlp = self._linear_lowp(
            f"{prefix}.txt_mlp.net.2",
            _gelu_approx(txt_mlp_hidden),
            out_dtype=self.residual_dtype,
        )
        txt = txt + txt_gate2[:, None, :] * txt_mlp

        return img, txt

    def forward(
        self,
        hidden_states: mx.array,
        *,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        img_shape: tuple[int, int, int],
        ref_latent_patches: list[mx.array] | None = None,
        ref_shapes: list[tuple[int, int, int]] | None = None,
        zero_cond_t: bool = False,
        increase_ref_index: bool = False,
        txt_pad_mask: mx.array | None = None,
    ) -> mx.array:
        hidden_states = self._linear_full(
            "img_in",
            hidden_states.astype(self.residual_dtype),
            out_dtype=self.residual_dtype,
        )
        main_img_seq = int(hidden_states.shape[1])
        batch = int(hidden_states.shape[0])

        if ref_latent_patches:
            ref_embeds = [
                self._linear_full(
                    "img_in",
                    ref.astype(self.residual_dtype),
                    out_dtype=self.residual_dtype,
                )
                for ref in ref_latent_patches
            ]
            if batch > 1:
                ref_embeds = [
                    mx.broadcast_to(r, (batch, r.shape[1], r.shape[2])) if r.shape[0] != batch else r
                    for r in ref_embeds
                ]
            hidden_states = mx.concatenate([hidden_states] + ref_embeds, axis=1)

        encoder_hidden_states = _rms_norm(
            encoder_hidden_states.astype(self.residual_dtype),
            self._weight("txt_norm.weight"),
            eps=self.q_eps,
        )
        encoder_hidden_states = self._linear_full(
            "txt_in",
            encoder_hidden_states,
            out_dtype=self.residual_dtype,
        )

        real_temb = self._time_embedding(timestep.astype(self.residual_dtype))
        if zero_cond_t:
            zero_steps = mx.zeros_like(timestep).astype(self.residual_dtype)
            zero_temb = self._time_embedding(zero_steps)
            temb = mx.concatenate([real_temb, zero_temb], axis=0)
        else:
            temb = real_temb

        modulate_index = None
        if zero_cond_t:
            total_img_seq = int(hidden_states.shape[1])
            num_ref_tokens = total_img_seq - main_img_seq
            modulate_index = mx.concatenate(
                [
                    mx.zeros((main_img_seq,), dtype=self.residual_dtype),
                    mx.ones((num_ref_tokens,), dtype=self.residual_dtype),
                ],
                axis=0,
            )

        if ref_shapes:
            img_rope, txt_rope = _build_qwen_image_edit_rope(
                img_shape=img_shape,
                ref_shapes=ref_shapes,
                txt_seq_len=int(encoder_hidden_states.shape[1]),
                increase_ref_index=increase_ref_index,
            )
        else:
            img_rope, txt_rope = _build_qwen_image_rope(
                img_shape=img_shape,
                txt_seq_len=int(encoder_hidden_states.shape[1]),
            )

        # Build additive SDPA mask once: shape [B, 1, 1, T_txt + T_img]. Padded
        # txt key positions get -inf so they don't leak into valid queries.
        attn_mask: mx.array | None = None
        if txt_pad_mask is not None:
            txt_len = int(encoder_hidden_states.shape[1])
            img_len = int(hidden_states.shape[1])
            ones_img = mx.ones((batch, img_len), dtype=txt_pad_mask.dtype)
            full_mask = mx.concatenate([txt_pad_mask, ones_img], axis=1)  # [B, T_total]
            neg_inf = mx.full(full_mask.shape, -1e4, dtype=self.attention_dtype)
            zero = mx.zeros(full_mask.shape, dtype=self.attention_dtype)
            additive = mx.where(full_mask > 0, zero, neg_inf)  # 0 for valid, -1e4 for pad
            attn_mask = additive[:, None, None, :]  # [B, 1, 1, T_total]

        for idx in range(self.num_layers):
            hidden_states, encoder_hidden_states = self._block(
                idx,
                hidden_states,
                encoder_hidden_states,
                temb,
                img_rope=img_rope,
                txt_rope=txt_rope,
                modulate_index=modulate_index,
                attn_mask=attn_mask,
            )

        out_mod = self._linear_full(
            "norm_out.linear",
            _silu(real_temb),
            out_dtype=self.residual_dtype,
        )
        out_scale, out_shift = mx.split(out_mod, 2, axis=-1)
        hidden_states = hidden_states[:, :main_img_seq, :]
        hidden_states = _layer_norm(hidden_states, eps=self.q_eps)
        hidden_states = hidden_states * (1.0 + out_scale[:, None, :]) + out_shift[:, None, :]
        return self._linear_full(
            "proj_out",
            hidden_states,
            out_dtype=self.residual_dtype,
        )


class QwenImageVaeDecoder:
    def __init__(self, weights: dict[str, mx.array]):
        self.w = weights
        self.compute_dtype = mx.float32

    def _weight(self, key: str) -> mx.array:
        if key not in self.w:
            raise KeyError(f"Missing weight: {key}")
        return self.w[key]

    def _conv3d(
        self,
        prefix: str,
        x: mx.array,
        *,
        padding: tuple[int, int, int] = (0, 0, 0),
        stride: tuple[int, int, int] = (1, 1, 1),
    ) -> mx.array:
        return _causal_conv3d_ncthw(
            x,
            self._weight(f"{prefix}.weight").astype(x.dtype),
            (
                self.w.get(f"{prefix}.bias").astype(x.dtype)
                if self.w.get(f"{prefix}.bias") is not None
                else None
            ),
            padding=padding,
            stride=stride,
        )

    def _conv2d(
        self,
        prefix: str,
        x: mx.array,
        *,
        padding: tuple[int, int] = (0, 0),
        stride: tuple[int, int] = (1, 1),
    ) -> mx.array:
        return _conv2d_nchw(
            x,
            self._weight(f"{prefix}.weight").astype(x.dtype),
            (
                self.w.get(f"{prefix}.bias").astype(x.dtype)
                if self.w.get(f"{prefix}.bias") is not None
                else None
            ),
            padding=padding,
            stride=stride,
        )

    def _residual_block(self, prefix: str, x: mx.array) -> mx.array:
        shortcut_key = f"{prefix}.shortcut.weight"
        h = self._conv3d(f"{prefix}.shortcut", x) if shortcut_key in self.w else x
        y = _rms_norm_channel_first(x, self._weight(f"{prefix}.residual.0.gamma"))
        y = _silu(y)
        y = self._conv3d(f"{prefix}.residual.2", y, padding=(1, 1, 1))
        y = _rms_norm_channel_first(y, self._weight(f"{prefix}.residual.3.gamma"))
        y = _silu(y)
        y = self._conv3d(f"{prefix}.residual.6", y, padding=(1, 1, 1))
        return h + y

    def _attention_block(self, prefix: str, x: mx.array) -> mx.array:
        identity = x
        x2, batch, time = _flatten_video_frames(x)
        x2 = _rms_norm_channel_first(x2, self._weight(f"{prefix}.norm.gamma"))
        qkv = self._conv2d(f"{prefix}.to_qkv", x2)
        batch_time, channels3, height, width = qkv.shape
        channels = channels3 // 3
        qkv = qkv.reshape((batch_time, 1, channels3, height * width)).transpose((0, 1, 3, 2))
        q, k, v = mx.split(qkv, 3, axis=-1)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(channels))
        out = out.squeeze(1).transpose((0, 2, 1)).reshape((batch_time, channels, height, width))
        out = self._conv2d(f"{prefix}.proj", out)
        return identity + _restore_video_frames(out, batch, time)

    def _resample(self, prefix: str, x: mx.array, *, temporal: bool) -> mx.array:
        if temporal and int(x.shape[2]) != 1:
            raise NotImplementedError("This VAE decoder only supports single-frame image decoding.")
        x = _upsample_spatial_2x_ncthw(x)
        x2, batch, time = _flatten_video_frames(x)
        x2 = self._conv2d(f"{prefix}.resample.1", x2, padding=(1, 1))
        return _restore_video_frames(x2, batch, time)

    def decode(self, z: mx.array) -> mx.array:
        if int(z.shape[2]) != 1:
            raise NotImplementedError("Only single-frame latent decoding is implemented.")

        x = z.astype(self.compute_dtype)
        x = self._conv3d("conv2", x)
        x = self._conv3d("decoder.conv1", x, padding=(1, 1, 1))

        x = self._residual_block("decoder.middle.0", x)
        x = self._attention_block("decoder.middle.1", x)
        x = self._residual_block("decoder.middle.2", x)

        for idx in range(3):
            x = self._residual_block(f"decoder.upsamples.{idx}", x)
        x = self._resample("decoder.upsamples.3", x, temporal=True)

        for idx in range(4, 7):
            x = self._residual_block(f"decoder.upsamples.{idx}", x)
        x = self._resample("decoder.upsamples.7", x, temporal=True)

        for idx in range(8, 11):
            x = self._residual_block(f"decoder.upsamples.{idx}", x)
        x = self._resample("decoder.upsamples.11", x, temporal=False)

        for idx in range(12, 15):
            x = self._residual_block(f"decoder.upsamples.{idx}", x)

        x = _rms_norm_channel_first(x, self._weight("decoder.head.0.gamma"))
        x = _silu(x)
        x = self._conv3d("decoder.head.2", x, padding=(1, 1, 1))
        return mx.clip(x, -1.0, 1.0)


class QwenImageVaeEncoder(QwenImageVaeDecoder):
    """Wan 2.1 VAE encoder for single-frame images (mirror of `QwenImageVaeDecoder`)."""

    def _downsample(self, prefix: str, x: mx.array) -> mx.array:
        # Spatial stride-2 Conv2d with symmetric pad=1, applied per frame.
        # `downsample3d` layers also contain `time_conv`, but for single-frame
        # input with no feature cache those are skipped (see sd.cpp `Resample`).
        if int(x.shape[2]) != 1:
            raise NotImplementedError("VAE encoder only supports single-frame inputs.")
        x2, batch, time = _flatten_video_frames(x)
        x2 = self._conv2d(
            f"{prefix}.resample.1",
            x2,
            padding=(1, 1),
            stride=(2, 2),
        )
        return _restore_video_frames(x2, batch, time)

    def encode(self, image: mx.array) -> mx.array:
        # image: [N, 3, 1, H, W] in [-1, 1]; returns raw mu [N, 16, 1, H/8, W/8].
        if int(image.shape[2]) != 1:
            raise NotImplementedError("Only single-frame encoding is implemented.")
        x = image.astype(self.compute_dtype)
        x = self._conv3d("encoder.conv1", x, padding=(1, 1, 1))

        x = self._residual_block("encoder.downsamples.0", x)
        x = self._residual_block("encoder.downsamples.1", x)
        x = self._downsample("encoder.downsamples.2", x)

        x = self._residual_block("encoder.downsamples.3", x)
        x = self._residual_block("encoder.downsamples.4", x)
        x = self._downsample("encoder.downsamples.5", x)

        x = self._residual_block("encoder.downsamples.6", x)
        x = self._residual_block("encoder.downsamples.7", x)
        x = self._downsample("encoder.downsamples.8", x)

        x = self._residual_block("encoder.downsamples.9", x)
        x = self._residual_block("encoder.downsamples.10", x)

        x = self._residual_block("encoder.middle.0", x)
        x = self._attention_block("encoder.middle.1", x)
        x = self._residual_block("encoder.middle.2", x)

        x = _rms_norm_channel_first(x, self._weight("encoder.head.0.gamma"))
        x = _silu(x)
        x = self._conv3d("encoder.head.2", x, padding=(1, 1, 1))

        # Post-encoder 1x1x1 conv: 32 -> 32 channels (produces mu + logvar split).
        x = self._conv3d("conv1", x)
        # Take first 16 channels as mu (deterministic encode).
        return x[:, :16, :, :, :]


def _resize_for_edit(
    image: Image.Image,
    *,
    target_width: int,
    target_height: int,
    vae_scale: int = 8,
    patch_size: int = 2,
    max_area: int = 1024 * 1024,
) -> Image.Image:
    """Match sd.cpp auto_resize_ref_image: preserve ref aspect, area=min(max_area, target_area)."""
    factor = vae_scale * patch_size * 2  # 32 for qwen-image
    w, h = image.size
    if h <= 0 or w <= 0:
        return image
    vae_image_size = min(max_area, target_width * target_height)
    vae_width = math.sqrt(vae_image_size * w / h)
    vae_height = vae_width * h / w
    new_w = int(round(vae_width / factor) * factor)
    new_h = int(round(vae_height / factor) * factor)
    new_w = max(factor, new_w)
    new_h = max(factor, new_h)
    if (new_w, new_h) != image.size:
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image


def _image_to_model_input(image: Image.Image) -> mx.array:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0  # [H, W, 3] in [-1, 1]
    arr = arr.transpose(2, 0, 1)[None, :, None, :, :]  # [1, 3, 1, H, W]
    return mx.array(arr)


def _smart_resize_vl(
    height: int,
    width: int,
    *,
    factor: int = 28,
    min_pixels: int = 384 * 384,
    max_pixels: int = 560 * 560,
) -> tuple[int, int]:
    """Qwen2.5-VL smart_resize: snap to factor, clamp area to [min_pixels, max_pixels]."""
    if min_pixels > max_pixels:
        raise ValueError("min_pixels must be <= max_pixels")
    h = max(factor, int(round(height / factor)) * factor)
    w = max(factor, int(round(width / factor)) * factor)
    if h * w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h = max(factor, int(math.floor(height / beta / factor)) * factor)
        w = max(factor, int(math.floor(width / beta / factor)) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = max(factor, int(math.ceil(height * beta / factor)) * factor)
        w = max(factor, int(math.ceil(width * beta / factor)) * factor)
    return h, w


def _preprocess_vision_image(
    image: Image.Image,
    *,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    min_pixels: int = 384 * 384,
    max_pixels: int = 560 * 560,
) -> tuple[mx.array, tuple[int, int, int]]:
    """
    Pre-process a single RGB image for the Qwen2.5-VL vision tower.

    Returns:
        pixel_values : mx.array, shape [T*grid_h*grid_w, in_channels * T * pH * pW].
        grid_thw    : (T, grid_h, grid_w) where T == temporal_patch_size after
                       single-frame duplication, so the tower sees a valid window.
    """
    if patch_size <= 0 or temporal_patch_size <= 0 or merge_size <= 0:
        raise ValueError("patch/temporal/merge sizes must be positive.")
    factor = patch_size * merge_size  # 28 by default
    w, h = image.size
    new_h, new_w = _smart_resize_vl(
        height=h,
        width=w,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if (new_w, new_h) != (w, h):
        image = image.resize((new_w, new_h), Image.BICUBIC)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0  # [H, W, 3]
    mean = np.array(QWEN_VL_IMAGE_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(QWEN_VL_IMAGE_STD, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # [3, H, W]
    # Duplicate the frame along T=temporal_patch_size so the tower sees a full window.
    patches = np.tile(arr[None, ...], (temporal_patch_size, 1, 1, 1))  # [T, 3, H, W]
    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h = new_h // patch_size
    grid_w = new_w // patch_size
    # reshape to [grid_t, T, C, grid_h, merge, pH, grid_w, merge, pW]
    patches = patches.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    # -> [grid_t, grid_h_m, grid_w_m, merge, merge, C, T, pH, pW]
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = patches.reshape(
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size,
    )
    return mx.array(flat), (grid_t, grid_h, grid_w)


def _load_qwen25_vl_vision_tower(
    safetensors_path: Path,
    *,
    verbose: bool = False,
):
    """Build mflux's VisionTransformer and populate `visual.*` weights in-place.

    The safetensors file is a Comfy-Org all-in-one Qwen2.5-VL checkpoint; we stream
    only the `visual.*` tensors. `patch_embed.proj.weight` is transposed from PyTorch's
    `[C_out, C_in, T, pH, pW]` layout into MLX Conv3d's `[C_out, T, pH, pW, C_in]`.
    """
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "safetensors is required to load the Qwen2.5-VL vision tower. "
            "Install with `pip install safetensors`."
        ) from exc

    try:
        import torch

        from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_transformer import (
            VisionTransformer,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Loading the vision tower needs torch + mflux. Install them before using "
            "`--reference-image` with the 2511 edit pipeline."
        ) from exc

    vt = VisionTransformer()
    params = vt.parameters()  # nested dict/list of mx.array leaves

    def _set_nested(tree, path: list[str], value: mx.array) -> None:
        node = tree
        for part in path[:-1]:
            if isinstance(node, list):
                node = node[int(part)]
            else:
                node = node[part]
        leaf = path[-1]
        if isinstance(node, list):
            node[int(leaf)] = value
        else:
            node[leaf] = value

    loaded_count = 0
    skipped = 0
    with safe_open(str(safetensors_path), framework="pt") as f:  # type: ignore
        for full_key in f.keys():  # type: ignore
            if not full_key.startswith("visual."):
                skipped += 1
                continue
            mapped = (
                full_key[len("visual.") :]
                .replace("merger.mlp.0.", "merger.mlp_0.")
                .replace("merger.mlp.2.", "merger.mlp_1.")
            )
            tensor = f.get_tensor(full_key)  # type: ignore
            if tensor.dtype in (torch.bfloat16, torch.float16):
                arr_np = tensor.to(torch.float32).numpy()
            else:
                arr_np = tensor.numpy()
            value = mx.array(arr_np)
            if mapped == "patch_embed.proj.weight":
                value = value.transpose(0, 2, 3, 4, 1)
            try:
                _set_nested(params, mapped.split("."), value)
            except (KeyError, IndexError) as exc:
                raise KeyError(
                    f"Vision-tower key {full_key!r} maps to {mapped!r}, which is not "
                    f"present in mflux VisionTransformer parameter tree: {exc}"
                )
            loaded_count += 1

    vt.update(params)
    if verbose:
        print(
            f"[vision] loaded {loaded_count} tensors from {safetensors_path.name} "
            f"(skipped {skipped} non-visual keys)"
        )
    return vt


def _run_vision_tower(
    vision_tower,
    image: Image.Image,
    *,
    min_pixels: int = 384 * 384,
    max_pixels: int = 560 * 560,
    verbose: bool = False,
) -> tuple[mx.array, tuple[int, int, int]]:
    """Run preprocess + vision tower; returns (image_embeds, grid_thw).

    The returned `image_embeds` already contains `(grid_h/merge * grid_w/merge)`
    tokens per frame (the tower's merger collapses the 2x2 spatial group), which
    is exactly how many `<|image_pad|>` placeholders the LLM prompt should carry.
    """
    pixel_values, grid_thw = _preprocess_vision_image(
        image,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    grid_thw_mx = mx.array([list(grid_thw)], dtype=mx.int32)
    if verbose:
        print(
            f"[vision] pixel_values={tuple(pixel_values.shape)} grid_thw={grid_thw} "
            f"(min_pixels={min_pixels}, max_pixels={max_pixels})"
        )
    embeds = vision_tower(pixel_values, grid_thw_mx)
    mx.eval(embeds)
    if verbose:
        _print_stats("vision.embeds", embeds, enabled=True)
    return embeds, grid_thw


def _encode_reference_latent(
    encoder: "QwenImageVaeEncoder",
    image: mx.array,
    *,
    verbose: bool = False,
) -> mx.array:
    raw = encoder.encode(image)  # [1, 16, 1, H/8, W/8]
    mean = mx.array(QWEN_IMAGE_LATENTS_MEAN, dtype=raw.dtype).reshape((1, 16, 1, 1, 1))
    std = mx.array(QWEN_IMAGE_LATENTS_STD, dtype=raw.dtype).reshape((1, 16, 1, 1, 1))
    normalized = (raw - mean) / std
    if verbose:
        _print_stats("ref_latent.raw", raw, enabled=True)
        _print_stats("ref_latent.normalized", normalized, enabled=True)
    return normalized


def _encode_prompt_embeddings(
    prompt: str,
    *,
    encoder: Qwen25VLTextEncoder,
    tokenizer: Any,
    tokenizer_source: str,
    max_sequence_length: int,
    verbose: bool,
    vision_embeds: list[mx.array] | None = None,
    vision_grids: list[tuple[int, int, int]] | None = None,
) -> mx.array:
    """Encode a prompt into T5-style prompt embeddings.

    If `vision_embeds` is provided, the edit-2511 vision prompt template is used:
      system prompt + "Picture N: <|vision_start|> <|image_pad|>*K <|vision_end|>" per ref
      + user text + assistant open tag. Image-pad embeddings are then overwritten with
      the vision tower's output before the LLM encoder runs, and the first
      EDIT_PROMPT_TEMPLATE_DROP_IDX tokens are dropped afterwards.

    Otherwise the original non-vision T2I template is used (drop_idx=34).
    """
    if vision_embeds:
        if vision_grids is None or len(vision_grids) != len(vision_embeds):
            raise ValueError("vision_embeds/vision_grids must be provided together and match length.")
        if tokenizer_source != "hf":
            raise ValueError(
                "Vision path requires --tokenizer to resolve to the HF Qwen2.5-VL tokenizer "
                "so that <|vision_start|>/<|image_pad|>/<|vision_end|> resolve to their "
                "single-token ids (151652/151655/151653)."
            )

        # Build the edit prompt: system + "Picture i: <|vision_start|>...<|vision_end|>\n" per ref + user.
        image_blocks: list[str] = []
        image_pad_counts: list[int] = []
        for idx, (embed, grid) in enumerate(zip(vision_embeds, vision_grids)):
            num_pad = int(embed.shape[0])
            if num_pad <= 0:
                raise ValueError(f"Vision embedding #{idx} has empty first dim.")
            image_pad_counts.append(num_pad)
            pad_str = "<|image_pad|>" * num_pad
            image_blocks.append(f"Picture {idx + 1}: <|vision_start|>{pad_str}<|vision_end|>")
        image_section = "".join(image_blocks)
        prompt_text = (
            EDIT_PROMPT_TEMPLATE_SYSTEM
            + image_section
            + prompt
            + EDIT_PROMPT_TEMPLATE_ASSISTANT
        )

        max_len = max_sequence_length + EDIT_PROMPT_TEMPLATE_DROP_IDX + sum(image_pad_counts) + 32
        token_ids = _tokenize_prompt(tokenizer, prompt_text, max_length=max_len, source=tokenizer_source)
        if not token_ids:
            raise ValueError("Prompt tokenization returned no tokens.")

        token_arr = mx.array([token_ids], dtype=mx.int32)
        inputs_embeds = encoder.embed_tokens(token_arr)  # [1, T, D]

        ids_np = np.array(token_ids, dtype=np.int64)
        pad_positions = np.where(ids_np == QWEN_VL_IMAGE_PAD_ID)[0]
        expected_pad = sum(image_pad_counts)
        if int(pad_positions.shape[0]) != expected_pad:
            raise ValueError(
                "Image-pad token count mismatch: tokenizer produced "
                f"{int(pad_positions.shape[0])} <|image_pad|> tokens but vision tower emitted "
                f"{expected_pad} tokens. Check tokenizer and vision preprocessing."
            )

        vision_cat = mx.concatenate([e.astype(inputs_embeds.dtype) for e in vision_embeds], axis=0)
        # Scatter the vision embeddings into the inputs_embeds tensor at the pad positions.
        pad_positions_mx = mx.array(pad_positions.astype(np.int32))
        flat = inputs_embeds.reshape((inputs_embeds.shape[1], inputs_embeds.shape[2]))
        flat = flat.at[pad_positions_mx].add(vision_cat - flat[pad_positions_mx])
        inputs_embeds = flat.reshape((1, inputs_embeds.shape[1], inputs_embeds.shape[2]))

        hidden = encoder.encode_last_hidden_state(inputs_embeds=inputs_embeds)
        drop_idx = EDIT_PROMPT_TEMPLATE_DROP_IDX
        hidden = hidden[:, drop_idx:, :]
        hidden = hidden[:, :max_sequence_length, :]
        if hidden.shape[1] <= 0:
            raise ValueError(
                "Prompt embedding sequence is empty after dropping the vision-edit template prefix."
            )
        if verbose:
            print(
                f"[text] vision-edit tokens={len(token_ids)} "
                f"pad_positions={len(pad_positions)} prompt_embeds={tuple(hidden.shape)}"
            )
        return hidden

    prompt_text = PROMPT_TEMPLATE.format(prompt)
    token_ids = _tokenize_prompt(
        tokenizer,
        prompt_text,
        max_length=max_sequence_length + PROMPT_TEMPLATE_DROP_IDX,
        source=tokenizer_source,
    )
    if not token_ids:
        raise ValueError("Prompt tokenization returned no tokens.")
    token_arr = mx.array([token_ids], dtype=mx.int32)
    hidden = encoder.encode_last_hidden_state(token_arr)
    hidden = hidden[:, PROMPT_TEMPLATE_DROP_IDX:, :]
    hidden = hidden[:, :max_sequence_length, :]
    if hidden.shape[1] <= 0:
        raise ValueError("Prompt embedding sequence is empty after dropping the fixed Qwen-Image template prefix.")
    if verbose:
        print(f"[text] tokens={len(token_ids)} prompt_embeds={tuple(hidden.shape)}")
    return hidden


def _generate_latents(
    *,
    transformer: QwenImageTransformer,
    prompt_embeds: mx.array,
    negative_prompt_embeds: mx.array | None,
    width: int,
    height: int,
    steps: int,
    seed: int,
    cfg_scale: float,
    verbose: bool,
    ref_latent_patches: list[mx.array] | None = None,
    ref_shapes: list[tuple[int, int, int]] | None = None,
    zero_cond_t: bool = False,
    increase_ref_index: bool = False,
    flow_shift: float | None = None,
    true_cfg_rescale: bool = False,
) -> mx.array:
    residual_dtype = getattr(transformer, "residual_dtype", transformer.compute_dtype)
    latent_h = height // 8
    latent_w = width // 8
    noise = _sample_latent_noise(
        seed=seed,
        height=latent_h,
        width=latent_w,
        channels=transformer.out_channels,
        dtype=residual_dtype,
    )
    latents = _pack_latents(noise)
    img_shape = (1, latent_h // 2, latent_w // 2)
    if flow_shift is not None:
        sigmas = _sigma_schedule_flow_shift(steps, flow_shift)
    else:
        sigmas = _sigma_schedule(steps, int(latents.shape[1]))

    if verbose:
        print(
            f"[sample] seed={seed} packed_latents={tuple(latents.shape)} "
            f"img_shape={img_shape} steps={steps} zero_cond_t={zero_cond_t} "
            f"flow_shift={flow_shift} refs={0 if not ref_latent_patches else len(ref_latent_patches)}"
        )
        _print_stats("latents.init", latents, enabled=True)

    prompt_embeds = prompt_embeds.astype(residual_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.astype(residual_dtype)

    ref_patches_rt = None
    if ref_latent_patches:
        ref_patches_rt = [p.astype(residual_dtype) for p in ref_latent_patches]

    # P0: when CFG is active, pad cond/uncond txt to the same length, stack as
    # batch=2 and run a single forward per step. Halves the number of forward
    # passes (and the Q4_K_M weight reads, which are the bandwidth bottleneck).
    cfg_batched = (
        negative_prompt_embeds is not None
        and cfg_scale > 1.0
    )
    batched_prompt: mx.array | None = None
    txt_pad_mask: mx.array | None = None
    if cfg_batched:
        t_pos = int(prompt_embeds.shape[1])
        t_neg = int(negative_prompt_embeds.shape[1])
        t_max = max(t_pos, t_neg)
        def _pad_txt(x: mx.array, t: int) -> mx.array:
            if int(x.shape[1]) == t:
                return x
            pad = mx.zeros((x.shape[0], t - x.shape[1], x.shape[2]), dtype=x.dtype)
            return mx.concatenate([x, pad], axis=1)
        pos_padded = _pad_txt(prompt_embeds, t_max)
        neg_padded = _pad_txt(negative_prompt_embeds, t_max)
        batched_prompt = mx.concatenate([pos_padded, neg_padded], axis=0)  # [2, T, D]
        # mask[b, t] = 1 if valid txt token, 0 if padded.
        pos_mask = mx.concatenate(
            [mx.ones((t_pos,), dtype=residual_dtype), mx.zeros((t_max - t_pos,), dtype=residual_dtype)],
            axis=0,
        )
        neg_mask = mx.concatenate(
            [mx.ones((t_neg,), dtype=residual_dtype), mx.zeros((t_max - t_neg,), dtype=residual_dtype)],
            axis=0,
        )
        txt_pad_mask = mx.stack([pos_mask, neg_mask], axis=0)  # [2, T]
        if verbose:
            print(
                f"[cfg-batch] padded cond/uncond {t_pos}/{t_neg} -> {t_max} "
                f"prompt_batched={tuple(batched_prompt.shape)}"
            )

    step_times: list[float] = []
    forward_times: list[float] = []
    for step_idx, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        t_step_start = time.perf_counter()
        timestep = mx.full((1,), sigma, dtype=residual_dtype)
        if cfg_batched:
            # Replicate latents to batch=2 and run cond+uncond in a single forward.
            latents_b = mx.broadcast_to(latents, (2, latents.shape[1], latents.shape[2]))
            both_pred = transformer.forward(
                latents_b,
                encoder_hidden_states=batched_prompt,
                timestep=timestep,
                img_shape=img_shape,
                ref_latent_patches=ref_patches_rt,
                ref_shapes=ref_shapes,
                zero_cond_t=zero_cond_t,
                increase_ref_index=increase_ref_index,
                txt_pad_mask=txt_pad_mask,
            )
            t_fwd: float | None = None
            if verbose:
                mx.eval(both_pred)
                t_fwd = time.perf_counter() - t_step_start
                forward_times.append(t_fwd)
            cond_pred = both_pred[0:1]
            uncond_pred = both_pred[1:2]
            combined = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
            if true_cfg_rescale:
                cond_norm = _norm_last_dim(cond_pred)
                combined_norm = _norm_last_dim(combined)
                noise_pred = combined * (cond_norm / combined_norm)
            else:
                noise_pred = combined
            _print_stats(f"noise_pred.step_{step_idx + 1}", cond_pred, enabled=verbose)
        else:
            noise_pred = transformer.forward(
                latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_shape=img_shape,
                ref_latent_patches=ref_patches_rt,
                ref_shapes=ref_shapes,
                zero_cond_t=zero_cond_t,
                increase_ref_index=increase_ref_index,
            )
            t_fwd: float | None = None
            if verbose:
                mx.eval(noise_pred)
                t_fwd = time.perf_counter() - t_step_start
                forward_times.append(t_fwd)
            _print_stats(f"noise_pred.step_{step_idx + 1}", noise_pred, enabled=verbose)

        latents = latents + (sigma_next - sigma) * noise_pred
        mx.eval(latents)
        t_step = time.perf_counter() - t_step_start
        step_times.append(t_step)

        extra = ""
        if t_fwd is not None:
            extra += f" fwd={t_fwd:.2f}s"
        avg_so_far = sum(step_times) / len(step_times)
        remaining = max(0, steps - (step_idx + 1))
        eta = remaining * avg_so_far
        print(
            f"[step {step_idx + 1:02d}/{steps}] sigma={sigma:.6f} -> {sigma_next:.6f} "
            f"step_time={t_step:.2f}s avg={avg_so_far:.2f}s eta={eta:.0f}s{extra}"
        )
        if verbose:
            _print_stats(f"latents.step_{step_idx + 1}", latents, enabled=True)

    if step_times:
        total = sum(step_times)
        avg = total / len(step_times)
        mn_t, mx_t = min(step_times), max(step_times)
        sorted_times = sorted(step_times)
        med = sorted_times[len(sorted_times) // 2]
        warm_avg = (
            sum(step_times[1:]) / (len(step_times) - 1) if len(step_times) >= 2 else avg
        )
        summary = (
            f"[sample-summary] steps={len(step_times)} total={total:.1f}s "
            f"avg={avg:.2f}s warm_avg={warm_avg:.2f}s "
            f"min={mn_t:.2f}s max={mx_t:.2f}s median={med:.2f}s"
        )
        if forward_times:
            summary += f" fwd_avg={sum(forward_times) / len(forward_times):.2f}s"
        if cfg_batched:
            summary += " cfg=batched"
        print(summary)

    return latents


def _save_image(image: mx.array, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_image_to_uint8(image)).save(output_path)


# Ordered list of pipeline stages. Stages that never ran (e.g. vision when no
# reference image was passed) are simply absent from `timings` and skipped.
_TIMING_STAGES: tuple[str, ...] = (
    "vision",
    "text_encode",
    "ref_vae_encode",
    "diffusion_load",
    "sampling",
    "vae_decode",
)


def _print_timing_summary(
    timings: dict[str, float],
    *,
    total: float,
    steps: int | None = None,
) -> None:
    label_width = max(len(s) for s in _TIMING_STAGES)
    accounted = 0.0
    lines = ["[timing-summary]"]
    for stage in _TIMING_STAGES:
        if stage not in timings:
            continue
        t = timings[stage]
        accounted += t
        extra = ""
        if stage == "sampling" and steps and steps > 0:
            extra = f"  ({steps} steps, avg {t / steps:.2f}s/step)"
        lines.append(f"  {stage.ljust(label_width)} : {t:7.2f}s{extra}")
    other = max(0.0, total - accounted)
    lines.append(f"  {'other'.ljust(label_width)} : {other:7.2f}s")
    lines.append(f"  {'-' * (label_width + 12)}")
    lines.append(f"  {'total'.ljust(label_width)} : {total:7.2f}s")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen-Image-Edit-2511 inference in MLX. Mirrors the sd.cpp flow: "
            "VAE-encode a reference image, concat ref patches along the seq dim, "
            "and run the diffusion transformer with dual (real/zero) time-embedding."
        )
    )
    parser.add_argument(
        "--diffusion-model",
        type=str,
        default="~/Downloads/qwen-image-edit-2511-Q4_K_M.gguf",
        help="Path to the Qwen-Image-Edit diffusion GGUF.",
    )
    parser.add_argument(
        "--vae",
        type=str,
        default="~/Downloads/qwen_image_vae.safetensors",
        help="Path to the Qwen-Image VAE safetensors.",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="~/Downloads/Qwen2.5-VL-7B-Instruct-Q4_0.gguf",
        help="Path to the Qwen2.5-VL GGUF text encoder.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help=(
            "Optional HF tokenizer repo/path. If omitted, the script first tries to "
            "guess a sibling HF tokenizer directory, then falls back to GGUF metadata."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading --tokenizer.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="Positive prompt.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Optional negative prompt used when --cfg-scale > 1.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=2.5,
        help="True CFG scale. Only applied when --negative-prompt is also provided.",
    )
    parser.add_argument(
        "-r",
        "--reference-image",
        type=str,
        default=None,
        help="Path to a reference image used as edit conditioning (sd.cpp -r).",
    )
    parser.add_argument(
        "--qwen-image-zero-cond-t",
        action="store_true",
        help=(
            "Enable Qwen-Image-Edit dual time-embedding (real-t for generated tokens, "
            "zero-t for ref tokens). Required for the Edit-2511 checkpoint."
        ),
    )
    parser.add_argument(
        "--flow-shift",
        type=float,
        default=3.0,
        help=(
            "Flow-matching sigma shift used by sd.cpp for Qwen-Image-Edit (mu = log(flow_shift)). "
            "Set to <=0 to fall back to dynamic resolution-based shift."
        ),
    )
    parser.add_argument(
        "--increase-ref-index",
        action="store_true",
        help="Give each reference a distinct RoPE index; otherwise a single shared index is used.",
    )
    parser.add_argument(
        "--max-ref-pixels",
        type=int,
        default=1024 * 1024,
        help="Upper bound for ref image area in pixels (sd.cpp: min(max_ref_pixels, target_w*target_h)).",
    )
    parser.add_argument(
        "--vision-tower",
        type=str,
        default=None,
        help=(
            "Path to qwen_2.5_vl_7b.safetensors (all-in-one LLM+vision) used only to load "
            "the vision tower (`visual.*`). When provided together with -r/--reference-image "
            "this enables the edit-2511 vision prompt path. Required for the 2511 checkpoint."
        ),
    )
    parser.add_argument(
        "--vision-min-pixels",
        type=int,
        default=384 * 384,
        help="Lower bound for vision-tower image area in pixels.",
    )
    parser.add_argument(
        "--vision-max-pixels",
        type=int,
        default=560 * 560,
        help="Upper bound for vision-tower image area in pixels.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output width in pixels. Rounded down to a multiple of 16.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output height in pixels. Rounded down to a multiple of 16.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for latent noise.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=1024,
        help="Maximum post-template prompt token length.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/qwen_image_edit.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution.",
    )
    parser.add_argument(
        "--offload-to-cpu",
        action="store_true",
        help="Alias for --cpu in this MLX implementation.",
    )
    parser.add_argument(
        "--diffusion-fa",
        action="store_true",
        help="Accepted for sd.cpp CLI compatibility. Ignored here.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print load and sampling details.",
    )
    args = parser.parse_args()

    if args.cpu or args.offload_to_cpu:
        mx.set_default_device(mx.cpu)

    if args.diffusion_fa and args.verbose:
        print("[warn] --diffusion-fa is accepted for CLI compatibility and ignored in this script.")
    if args.offload_to_cpu and args.verbose:
        print("[warn] --offload-to-cpu maps to whole-process CPU execution in MLX.")

    if args.steps <= 0:
        raise ValueError(f"--steps must be > 0, got {args.steps}.")
    if args.max_sequence_length <= 0 or args.max_sequence_length > 1024:
        raise ValueError("--max-sequence-length must be in [1, 1024].")

    width = max(16, (args.width // 16) * 16)
    height = max(16, (args.height // 16) * 16)
    if (width, height) != (args.width, args.height):
        print(
            f"[warn] resizing requested dimensions from {args.width}x{args.height} "
            f"to {width}x{height} so they are divisible by 16."
        )

    diffusion_path = Path(args.diffusion_model).expanduser()
    vae_path = Path(args.vae).expanduser()
    llm_path = Path(args.llm).expanduser()
    output_path = Path(args.output).expanduser()
    for path in (diffusion_path, vae_path, llm_path):
        if not path.exists():
            raise FileNotFoundError(f"Required model file not found: {path}")

    # Resolve vision path early so we know whether to force the HF tokenizer.
    vision_enabled = bool(args.reference_image and args.vision_tower)
    if args.reference_image and not args.vision_tower:
        print(
            "[warn] --reference-image provided without --vision-tower: running the "
            "non-vision edit path (LLM prompt has no ref-image embeddings). "
            "This is usually wrong for the 2511 checkpoint; pass "
            "--vision-tower ~/Downloads/qwen_2.5_vl_7b.safetensors."
        )
    if args.vision_tower and not args.reference_image:
        print("[warn] --vision-tower is set but --reference-image is missing; ignoring --vision-tower.")

    # Per-stage wall-clock timings. Always collected; printed as a consolidated
    # summary at the very end regardless of --verbose.
    timings: dict[str, float] = {}
    t_total = time.perf_counter()

    tokenizer_ref = args.tokenizer
    if vision_enabled and tokenizer_ref is None:
        tokenizer_ref = QWEN25_VL_HF_REPO
    if tokenizer_ref is None:
        tokenizer_ref = _guess_tokenizer_ref(llm_path)
    if args.verbose and tokenizer_ref is not None:
        print(f"[tokenizer] using {tokenizer_ref}")

    # --- Stage A: run vision tower on the reference image (if enabled). ---
    vision_embeds: list[mx.array] | None = None
    vision_grids: list[tuple[int, int, int]] | None = None
    ref_image_pil: Image.Image | None = None
    if vision_enabled:
        vt_path = Path(args.vision_tower).expanduser()
        if not vt_path.exists():
            raise FileNotFoundError(f"Vision tower safetensors not found: {vt_path}")
        ref_path = Path(args.reference_image).expanduser()
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image not found: {ref_path}")
        t_vision = time.perf_counter()
        ref_image_pil = Image.open(ref_path).convert("RGB")
        if args.verbose:
            print(f"[vision] loading tower from {vt_path.name}...")
        vision_tower = _load_qwen25_vl_vision_tower(vt_path, verbose=args.verbose)
        embed, grid = _run_vision_tower(
            vision_tower,
            ref_image_pil,
            min_pixels=args.vision_min_pixels,
            max_pixels=args.vision_max_pixels,
            verbose=args.verbose,
        )
        vision_embeds = [embed]
        vision_grids = [grid]
        timings["vision"] = time.perf_counter() - t_vision
        if args.verbose:
            print(f"[vision] finished in {timings['vision']:.2f}s")
        del vision_tower
        gc.collect()
        mx.clear_cache()

    # --- Stage B: load text encoder + tokenizer, encode prompts. ---
    t0 = time.perf_counter()
    llm_weights, llm_meta = _load_gguf(str(llm_path))
    tokenizer, tokenizer_source = _pick_tokenizer(
        llm_meta,
        tokenizer_ref=tokenizer_ref,
        trust_remote_code=args.trust_remote_code,
    )
    text_encoder = Qwen25VLTextEncoder(llm_weights, llm_meta)
    prompt_embeds = _encode_prompt_embeddings(
        args.prompt,
        encoder=text_encoder,
        tokenizer=tokenizer,
        tokenizer_source=tokenizer_source,
        max_sequence_length=args.max_sequence_length,
        verbose=args.verbose,
        vision_embeds=vision_embeds,
        vision_grids=vision_grids,
    )
    negative_prompt_embeds = None
    effective_negative_prompt = args.negative_prompt
    if effective_negative_prompt is None and args.cfg_scale > 1.0:
        # sd.cpp defaults the unconditional prompt to empty-string when cfg>1.
        effective_negative_prompt = ""
        if args.verbose:
            print("[cfg] --negative-prompt not provided; using empty-string uncond (sd.cpp parity).")
    if effective_negative_prompt is not None:
        # Negative prompt still uses the vision template when vision is enabled so the
        # drop_idx and the conditioning/uncond sequence shapes stay consistent.
        negative_prompt_embeds = _encode_prompt_embeddings(
            effective_negative_prompt,
            encoder=text_encoder,
            tokenizer=tokenizer,
            tokenizer_source=tokenizer_source,
            max_sequence_length=args.max_sequence_length,
            verbose=args.verbose,
            vision_embeds=vision_embeds,
            vision_grids=vision_grids,
        )
    if negative_prompt_embeds is None:
        mx.eval(prompt_embeds)
    else:
        mx.eval(prompt_embeds, negative_prompt_embeds)
    timings["text_encode"] = time.perf_counter() - t0
    if args.verbose:
        print(f"[load] text encoder + tokenizer ready in {timings['text_encode']:.2f}s")

    del text_encoder
    del llm_weights
    gc.collect()
    mx.clear_cache()

    # --- Stage C: VAE-encode the reference image (ref latent for transformer). ---
    ref_latent_patches: list[mx.array] | None = None
    ref_shapes: list[tuple[int, int, int]] | None = None
    if args.reference_image is not None:
        ref_path = Path(args.reference_image).expanduser()
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image not found: {ref_path}")
        t_ref = time.perf_counter()
        if ref_image_pil is None:
            ref_image_pil = Image.open(ref_path).convert("RGB")
        ref_image_pil_for_vae = _resize_for_edit(
            ref_image_pil,
            target_width=width,
            target_height=height,
            max_area=args.max_ref_pixels,
        )
        if args.verbose:
            print(f"[ref] loaded {ref_path.name} resized to {ref_image_pil_for_vae.size}")
        ref_model_input = _image_to_model_input(ref_image_pil_for_vae)

        vae_weights = _load_safetensors(str(vae_path))
        encoder = QwenImageVaeEncoder(vae_weights)
        ref_latent = _encode_reference_latent(encoder, ref_model_input, verbose=args.verbose)
        mx.eval(ref_latent)

        ref_latent_patches = [_pack_latents(ref_latent)]
        ref_latent_h = int(ref_latent.shape[3])
        ref_latent_w = int(ref_latent.shape[4])
        ref_shapes = [(1, ref_latent_h // 2, ref_latent_w // 2)]
        timings["ref_vae_encode"] = time.perf_counter() - t_ref
        if args.verbose:
            print(
                f"[ref] encoded latent=[1,16,1,{ref_latent_h},{ref_latent_w}] "
                f"packed={tuple(ref_latent_patches[0].shape)} "
                f"ref_shape={ref_shapes[0]} elapsed={timings['ref_vae_encode']:.2f}s"
            )

        del encoder
        del vae_weights
        gc.collect()
        mx.clear_cache()

    flow_shift = args.flow_shift if args.flow_shift and args.flow_shift > 0 else None

    t_diffusion_load = time.perf_counter()
    diffusion_weights, diffusion_meta = _load_gguf(str(diffusion_path))
    transformer = QwenImageTransformer(diffusion_weights, diffusion_meta)
    zero_cond_t_flag = args.qwen_image_zero_cond_t
    if "__index_timestep_zero__" in diffusion_weights and not zero_cond_t_flag:
        zero_cond_t_flag = True
        if args.verbose:
            print("[edit] auto-enabling zero_cond_t: GGUF contains __index_timestep_zero__ marker.")
    if args.verbose:
        print(
            f"[precision] transformer_mlp_lowp={transformer.mlp_lowp_dtype} "
            f"attention={transformer.attention_dtype} residual={transformer.residual_dtype}"
        )
    timings["diffusion_load"] = time.perf_counter() - t_diffusion_load
    if args.verbose:
        print(f"[load] diffusion model ready in {timings['diffusion_load']:.2f}s")

    t_sample = time.perf_counter()
    latents = _generate_latents(
        transformer=transformer,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        width=width,
        height=height,
        steps=args.steps,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        verbose=args.verbose,
        ref_latent_patches=ref_latent_patches,
        ref_shapes=ref_shapes,
        zero_cond_t=zero_cond_t_flag,
        increase_ref_index=args.increase_ref_index,
        flow_shift=flow_shift,
    )
    mx.eval(latents)
    timings["sampling"] = time.perf_counter() - t_sample
    if args.verbose:
        print(f"[sample] finished in {timings['sampling']:.2f}s")

    del transformer
    del diffusion_weights
    gc.collect()
    mx.clear_cache()

    t2 = time.perf_counter()
    latents = _unpack_latents(latents, height=height, width=width).astype(mx.float32)
    latents_mean = mx.array(QWEN_IMAGE_LATENTS_MEAN, dtype=latents.dtype).reshape((1, 16, 1, 1, 1))
    latents_std = mx.array(QWEN_IMAGE_LATENTS_STD, dtype=latents.dtype).reshape((1, 16, 1, 1, 1))
    latents = latents * latents_std + latents_mean
    _print_stats("latents.before_vae", latents, enabled=args.verbose)

    vae_weights = _load_safetensors(str(vae_path))
    vae = QwenImageVaeDecoder(vae_weights)
    decoded = vae.decode(latents)
    mx.eval(decoded)
    _print_stats("decoded", decoded, enabled=args.verbose)
    image = decoded[0, :, 0, :, :]
    _save_image(image, output_path)

    timings["vae_decode"] = time.perf_counter() - t2
    if args.verbose:
        print(f"[decode] vae finished in {timings['vae_decode']:.2f}s")

    _print_timing_summary(timings, total=time.perf_counter() - t_total, steps=args.steps)
    print(output_path)


if __name__ == "__main__":
    main()

