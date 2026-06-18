#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image

from infer_qwen_gguf import (
    GGUFQwen,
    apply_qwen_chat_template,
    build_tokenizer,
    resolve_affine_quant_params,
)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _layer_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return (x - mean) * mx.rsqrt(var + eps)


def _rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    x32 = x.astype(mx.float32)
    rrms = mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return (x32 * rrms).astype(x.dtype) * weight.astype(x.dtype)


def _linear(x: mx.array, weight: mx.array) -> mx.array:
    return x @ weight.T


def _conv_weight(weight: mx.array) -> mx.array:
    # PyTorch stores OIHW, MLX expects OHWI for NHWC inputs.
    return weight.transpose((0, 2, 3, 1))


def _conv2d(
    x: mx.array,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    stride: int = 1,
    padding: int = 0,
) -> mx.array:
    if weight.ndim == 2:
        if stride != 1 or padding != 0:
            raise ValueError(
                "2D pointwise weights only support stride=1 and padding=0, "
                f"got stride={stride}, padding={padding}."
            )
        y = x @ weight.T
        if bias is not None:
            y = y + bias
        return y
    y = mx.conv2d(
        x,
        _conv_weight(weight),
        stride=(stride, stride),
        padding=(padding, padding),
    )
    if bias is not None:
        y = y + bias
    return y


def _group_norm(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    *,
    num_groups: int = 32,
    eps: float = 1e-6,
) -> mx.array:
    batch, height, width, channels = x.shape
    group_size = channels // num_groups
    xg = x.reshape((batch, -1, num_groups, group_size))
    xg = xg.transpose((0, 2, 1, 3)).reshape((batch, num_groups, -1))
    mean = mx.mean(xg, axis=-1, keepdims=True)
    var = mx.var(xg, axis=-1, keepdims=True)
    xg = (xg - mean) * mx.rsqrt(var + eps)
    xg = xg.reshape((batch, num_groups, -1, group_size))
    xg = xg.transpose((0, 2, 1, 3)).reshape((batch, height, width, channels))
    return xg * weight + bias


def _upsample_nearest_2x(x: mx.array) -> mx.array:
    batch, height, width, channels = x.shape
    x = x.reshape((batch, height, 1, width, 1, channels))
    x = mx.broadcast_to(x, (batch, height, 2, width, 2, channels))
    return x.reshape((batch, height * 2, width * 2, channels))


def _reshape_for_heads(
    x: mx.array,
    *,
    num_heads: int,
    head_dim: int,
) -> mx.array:
    batch, seq_len, _ = x.shape
    return x.reshape((batch, seq_len, num_heads, head_dim)).transpose((0, 2, 1, 3))


def _merge_heads(x: mx.array) -> mx.array:
    batch, num_heads, seq_len, head_dim = x.shape
    return x.transpose((0, 2, 1, 3)).reshape((batch, seq_len, num_heads * head_dim))


def _rope_1d(pos: mx.array, dim: int, theta: int) -> mx.array:
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {dim}.")
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    omega = 1.0 / (theta**scale)
    out = pos.astype(mx.float32)[..., None] * omega
    out = mx.stack([mx.cos(out), -mx.sin(out), mx.sin(out), mx.cos(out)], axis=-1)
    return out.reshape((*out.shape[:-1], 2, 2))


def _embed_nd(ids: mx.array, axes_dim: tuple[int, ...], theta: int) -> mx.array:
    parts = [_rope_1d(ids[..., i], axes_dim[i], theta) for i in range(len(axes_dim))]
    return mx.concatenate(parts, axis=2)[:, None, ...]


def _apply_nd_rope(xq: mx.array, xk: mx.array, freqs: mx.array) -> tuple[mx.array, mx.array]:
    xq_ = xq.astype(mx.float32).reshape((*xq.shape[:-1], -1, 1, 2))
    xk_ = xk.astype(mx.float32).reshape((*xk.shape[:-1], -1, 1, 2))
    xq_out = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_out = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_out.reshape(xq.shape).astype(xq.dtype), xk_out.reshape(xk.shape).astype(xk.dtype)


def _qk_norm(
    q: mx.array,
    k: mx.array,
    q_scale: mx.array,
    k_scale: mx.array,
    *,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array]:
    return _rms_norm(q, q_scale, eps=eps), _rms_norm(k, k_scale, eps=eps)


def _attention(q: mx.array, k: mx.array, v: mx.array, *, scale: float) -> mx.array:
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)


def _modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return (1.0 + scale) * x + shift


def _silu_gate(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return _silu(x1) * x2


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
        -math.log(max_period)
        * mx.arange(0, half, dtype=mx.float32)
        / max(half, 1)
    )
    args = timesteps[:, None] * freqs[None, :]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1)
    return emb.astype(timesteps.dtype)


def _build_text_position_ids(batch_size: int, seq_len: int) -> mx.array:
    zeros = mx.zeros((seq_len,), dtype=mx.int32)
    ids = mx.stack([zeros, zeros, zeros, mx.arange(seq_len, dtype=mx.int32)], axis=-1)
    return mx.broadcast_to(ids[None, :, :], (batch_size, seq_len, 4))


def _build_image_position_ids(batch_size: int, height: int, width: int) -> mx.array:
    t_ids = mx.zeros((height * width,), dtype=mx.int32)
    h_ids = mx.broadcast_to(mx.arange(height, dtype=mx.int32)[:, None], (height, width)).reshape((-1,))
    w_ids = mx.broadcast_to(mx.arange(width, dtype=mx.int32)[None, :], (height, width)).reshape((-1,))
    l_ids = mx.zeros((height * width,), dtype=mx.int32)
    ids = mx.stack([t_ids, h_ids, w_ids, l_ids], axis=-1)
    return mx.broadcast_to(ids[None, :, :], (batch_size, height * width, 4))


def _flatten_image_tokens(x: mx.array) -> mx.array:
    batch, height, width, channels = x.shape
    return x.reshape((batch, height * width, channels))


def _unflatten_image_tokens(x: mx.array, height: int, width: int) -> mx.array:
    batch, seq_len, channels = x.shape
    expected = height * width
    if seq_len != expected:
        raise ValueError(f"Expected {expected} image tokens, got {seq_len}.")
    return x.reshape((batch, height, width, channels))


def _compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def _generalized_time_snr_shift(t: mx.array, mu: float, sigma: float) -> mx.array:
    numerator = math.exp(mu)
    return numerator / (numerator + (1.0 / t - 1.0) ** sigma)


def _get_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    mu = _compute_empirical_mu(image_seq_len, num_steps)
    timesteps = mx.linspace(1.0, 0.0, num_steps + 1)
    timesteps = _generalized_time_snr_shift(timesteps, mu, 1.0)
    return [float(x) for x in timesteps.tolist()]


def _count_indexed_submodules(weights: dict[str, mx.array], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    indices = {int(match.group(1)) for key in weights if (match := pattern.match(key))}
    if not indices:
        raise ValueError(f"Missing weights for {prefix}.")
    expected = set(range(max(indices) + 1))
    if indices != expected:
        raise ValueError(
            f"Non-contiguous {prefix} indices: expected {sorted(expected)}, got {sorted(indices)}."
        )
    return len(indices)


@dataclass(frozen=True)
class Flux2Config:
    in_channels: int
    context_in_dim: int
    hidden_size: int
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: tuple[int, int, int, int]
    theta: int = 2000
    mlp_ratio: float = 3.0
    use_guidance_embed: bool = False

    @classmethod
    def from_weights(cls, weights: dict[str, mx.array]) -> "Flux2Config":
        img_in = weights.get("img_in.weight")
        txt_in = weights.get("txt_in.weight")
        q_norm = weights.get("double_blocks.0.img_attn.norm.query_norm.scale")
        mlp_in = weights.get("double_blocks.0.img_mlp.0.weight")

        if img_in is None or txt_in is None or q_norm is None or mlp_in is None:
            raise ValueError(
                "Missing required Flux.2 tensors while inferring config. "
                "Expected img_in.weight, txt_in.weight, double_blocks.0.img_attn.norm.query_norm.scale, "
                "and double_blocks.0.img_mlp.0.weight."
            )

        hidden_size, in_channels = (int(x) for x in img_in.shape)
        txt_hidden_size, context_in_dim = (int(x) for x in txt_in.shape)
        if txt_hidden_size != hidden_size:
            raise ValueError(
                f"txt_in.weight output dim {txt_hidden_size} does not match img_in.weight output dim {hidden_size}."
            )

        head_dim = int(q_norm.shape[0])
        if head_dim <= 0 or hidden_size % head_dim != 0:
            raise ValueError(
                f"Cannot infer attention heads: hidden_size={hidden_size}, head_dim={head_dim}."
            )
        if head_dim % 4 != 0:
            raise ValueError(f"Expected head_dim divisible by 4 for ND RoPE, got {head_dim}.")

        mlp_ratio = int(mlp_in.shape[0]) / (2.0 * hidden_size)
        return cls(
            in_channels=in_channels,
            context_in_dim=context_in_dim,
            hidden_size=hidden_size,
            num_heads=hidden_size // head_dim,
            depth=_count_indexed_submodules(weights, "double_blocks"),
            depth_single_blocks=_count_indexed_submodules(weights, "single_blocks"),
            axes_dim=(head_dim // 4, head_dim // 4, head_dim // 4, head_dim // 4),
            mlp_ratio=mlp_ratio,
        )


class Flux2Model:
    def __init__(self, weights: dict[str, mx.array], cfg: Flux2Config):
        self.w = weights
        self.cfg = cfg
        self.out_channels = cfg.in_channels
        self.head_dim = cfg.hidden_size // cfg.num_heads
        if self.cfg.hidden_size % self.cfg.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if sum(self.cfg.axes_dim) != self.head_dim:
            raise ValueError(
                f"axes_dim={self.cfg.axes_dim} does not sum to head_dim={self.head_dim}."
            )
        self.mlp_hidden_dim = int(cfg.hidden_size * cfg.mlp_ratio)
        self.attn_scale = 1.0 / math.sqrt(self.head_dim)
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
        if self._has_quantized_triplet(prefix):
            if prefix not in self._quant_params_cache:
                self._quant_params_cache[prefix] = resolve_affine_quant_params(
                    self.w,
                    {},
                    prefix,
                )
            params = self._quant_params_cache[prefix]
            return mx.quantized_matmul(
                x,
                self._weight(f"{prefix}.weight"),
                self._weight(f"{prefix}.scales"),
                self._weight(f"{prefix}.biases"),
                transpose=True,
                group_size=params.group_size,
                bits=params.bits,
                mode=params.mode,
            )
        return _linear(x, self._weight(f"{prefix}.weight"))

    def _mlp_embedder(self, prefix: str, x: mx.array) -> mx.array:
        x = self._linear_prefix(f"{prefix}.in_layer", x)
        x = _silu(x)
        return self._linear_prefix(f"{prefix}.out_layer", x)

    def _modulation(
        self,
        prefix: str,
        vec: mx.array,
        *,
        double: bool,
    ) -> tuple[tuple[mx.array, mx.array, mx.array], tuple[mx.array, mx.array, mx.array] | None]:
        multiplier = 6 if double else 3
        out = self._linear_prefix(f"{prefix}.lin", _silu(vec))
        out = out[:, None, :]
        chunks = mx.split(out, multiplier, axis=-1)
        first = (chunks[0], chunks[1], chunks[2])
        second = (chunks[3], chunks[4], chunks[5]) if double else None
        return first, second

    def _double_block(
        self,
        idx: int,
        img: mx.array,
        txt: mx.array,
        pe_x: mx.array,
        pe_txt: mx.array,
        mod_img: tuple[tuple[mx.array, mx.array, mx.array], tuple[mx.array, mx.array, mx.array]],
        mod_txt: tuple[tuple[mx.array, mx.array, mx.array], tuple[mx.array, mx.array, mx.array]],
    ) -> tuple[mx.array, mx.array]:
        img_mod1, img_mod2 = mod_img
        txt_mod1, txt_mod2 = mod_txt

        img_modulated = _modulate(_layer_norm(img), img_mod1[0], img_mod1[1])
        txt_modulated = _modulate(_layer_norm(txt), txt_mod1[0], txt_mod1[1])

        img_qkv = self._linear_prefix(f"double_blocks.{idx}.img_attn.qkv", img_modulated)
        txt_qkv = self._linear_prefix(f"double_blocks.{idx}.txt_attn.qkv", txt_modulated)

        hidden = self.cfg.hidden_size
        img_q = _reshape_for_heads(img_qkv[..., :hidden], num_heads=self.cfg.num_heads, head_dim=self.head_dim)
        img_k = _reshape_for_heads(
            img_qkv[..., hidden : 2 * hidden],
            num_heads=self.cfg.num_heads,
            head_dim=self.head_dim,
        )
        img_v = _reshape_for_heads(
            img_qkv[..., 2 * hidden : 3 * hidden],
            num_heads=self.cfg.num_heads,
            head_dim=self.head_dim,
        )

        txt_q = _reshape_for_heads(txt_qkv[..., :hidden], num_heads=self.cfg.num_heads, head_dim=self.head_dim)
        txt_k = _reshape_for_heads(
            txt_qkv[..., hidden : 2 * hidden],
            num_heads=self.cfg.num_heads,
            head_dim=self.head_dim,
        )
        txt_v = _reshape_for_heads(
            txt_qkv[..., 2 * hidden : 3 * hidden],
            num_heads=self.cfg.num_heads,
            head_dim=self.head_dim,
        )

        img_q, img_k = _qk_norm(
            img_q,
            img_k,
            self._weight(f"double_blocks.{idx}.img_attn.norm.query_norm.scale"),
            self._weight(f"double_blocks.{idx}.img_attn.norm.key_norm.scale"),
        )
        txt_q, txt_k = _qk_norm(
            txt_q,
            txt_k,
            self._weight(f"double_blocks.{idx}.txt_attn.norm.query_norm.scale"),
            self._weight(f"double_blocks.{idx}.txt_attn.norm.key_norm.scale"),
        )

        q = mx.concatenate([txt_q, img_q], axis=2)
        k = mx.concatenate([txt_k, img_k], axis=2)
        v = mx.concatenate([txt_v, img_v], axis=2)
        pe_full = mx.concatenate([pe_txt, pe_x], axis=2)
        q, k = _apply_nd_rope(q, k, pe_full)
        attn = _attention(q, k, v, scale=self.attn_scale)
        attn = _merge_heads(attn)

        num_txt_tokens = txt.shape[1]
        txt_attn = attn[:, :num_txt_tokens, :]
        img_attn = attn[:, num_txt_tokens:, :]

        img = img + img_mod1[2] * self._linear_prefix(
            f"double_blocks.{idx}.img_attn.proj",
            img_attn,
        )
        img = img + img_mod2[2] * self._linear_prefix(
            f"double_blocks.{idx}.img_mlp.2",
            _silu_gate(
                self._linear_prefix(
                    f"double_blocks.{idx}.img_mlp.0",
                    _modulate(_layer_norm(img), img_mod2[0], img_mod2[1]),
                )
            ),
        )

        txt = txt + txt_mod1[2] * self._linear_prefix(
            f"double_blocks.{idx}.txt_attn.proj",
            txt_attn,
        )
        txt = txt + txt_mod2[2] * self._linear_prefix(
            f"double_blocks.{idx}.txt_mlp.2",
            _silu_gate(
                self._linear_prefix(
                    f"double_blocks.{idx}.txt_mlp.0",
                    _modulate(_layer_norm(txt), txt_mod2[0], txt_mod2[1]),
                )
            ),
        )
        return img, txt

    def _single_block(
        self,
        idx: int,
        x: mx.array,
        pe: mx.array,
        mod: tuple[mx.array, mx.array, mx.array],
    ) -> mx.array:
        x_mod = _modulate(_layer_norm(x), mod[0], mod[1])
        proj = self._linear_prefix(f"single_blocks.{idx}.linear1", x_mod)
        qkv = proj[..., : 3 * self.cfg.hidden_size]
        mlp = proj[..., 3 * self.cfg.hidden_size :]

        q = _reshape_for_heads(qkv[..., : self.cfg.hidden_size], num_heads=self.cfg.num_heads, head_dim=self.head_dim)
        k = _reshape_for_heads(
            qkv[..., self.cfg.hidden_size : 2 * self.cfg.hidden_size],
            num_heads=self.cfg.num_heads,
            head_dim=self.head_dim,
        )
        v = _reshape_for_heads(
            qkv[..., 2 * self.cfg.hidden_size : 3 * self.cfg.hidden_size],
            num_heads=self.cfg.num_heads,
            head_dim=self.head_dim,
        )
        q, k = _qk_norm(
            q,
            k,
            self._weight(f"single_blocks.{idx}.norm.query_norm.scale"),
            self._weight(f"single_blocks.{idx}.norm.key_norm.scale"),
        )
        q, k = _apply_nd_rope(q, k, pe)
        attn = _attention(q, k, v, scale=self.attn_scale)
        attn = _merge_heads(attn)
        out = self._linear_prefix(
            f"single_blocks.{idx}.linear2",
            mx.concatenate([attn, _silu_gate(mlp)], axis=-1),
        )
        return x + mod[2] * out

    def _final_layer(self, x: mx.array, vec: mx.array) -> mx.array:
        mod = self._linear_prefix("final_layer.adaLN_modulation.1", _silu(vec))
        shift, scale = mx.split(mod, 2, axis=-1)
        shift = shift[:, None, :]
        scale = scale[:, None, :]
        x = _modulate(_layer_norm(x), shift, scale)
        return self._linear_prefix("final_layer.linear", x)

    def forward(
        self,
        img: mx.array,
        img_ids: mx.array,
        txt: mx.array,
        txt_ids: mx.array,
        timesteps: mx.array,
        guidance: mx.array | None = None,
    ) -> mx.array:
        del guidance
        if img.shape[-1] != self.cfg.in_channels:
            raise ValueError(
                f"Expected image tokens with channel dim {self.cfg.in_channels}, got {img.shape[-1]}."
            )
        if txt.shape[-1] != self.cfg.context_in_dim:
            raise ValueError(
                f"Expected text context dim {self.cfg.context_in_dim}, got {txt.shape[-1]}."
            )
        vec = self._mlp_embedder("time_in", _timestep_embedding(timesteps, 256))
        mod_img = self._modulation("double_stream_modulation_img", vec, double=True)
        mod_txt = self._modulation("double_stream_modulation_txt", vec, double=True)
        mod_single = self._modulation("single_stream_modulation", vec, double=False)[0]

        img = self._linear_prefix("img_in", img)
        txt = self._linear_prefix("txt_in", txt)

        pe_x = _embed_nd(img_ids, self.cfg.axes_dim, self.cfg.theta)
        pe_txt = _embed_nd(txt_ids, self.cfg.axes_dim, self.cfg.theta)

        for i in range(self.cfg.depth):
            img, txt = self._double_block(
                i,
                img,
                txt,
                pe_x,
                pe_txt,
                mod_img=mod_img,
                mod_txt=mod_txt,
            )

        num_txt_tokens = txt.shape[1]
        x = mx.concatenate([txt, img], axis=1)
        pe = mx.concatenate([pe_txt, pe_x], axis=2)

        for i in range(self.cfg.depth_single_blocks):
            x = self._single_block(i, x, pe, mod_single)

        x = x[:, num_txt_tokens:, :]
        return self._final_layer(x, vec)


class Flux2VaeDecoder:
    def __init__(self, weights: dict[str, mx.array]):
        self.w = weights
        self.bn_eps = 1e-4
        self.pixel_shuffle = (2, 2)

    def _weight(self, key: str) -> mx.array:
        if key not in self.w:
            raise KeyError(f"Missing weight: {key}")
        return self.w[key]

    def _conv(self, prefix: str, x: mx.array, *, stride: int = 1, padding: int = 1) -> mx.array:
        weight = self._weight(f"{prefix}.weight")
        bias = self.w.get(f"{prefix}.bias")
        return _conv2d(x, weight, bias, stride=stride, padding=padding)

    def _resnet(self, prefix: str, x: mx.array) -> mx.array:
        h = _group_norm(
            x,
            self._weight(f"{prefix}.norm1.weight"),
            self._weight(f"{prefix}.norm1.bias"),
        )
        h = _silu(h)
        h = self._conv(f"{prefix}.conv1", h, padding=1)
        h = _group_norm(
            h,
            self._weight(f"{prefix}.norm2.weight"),
            self._weight(f"{prefix}.norm2.bias"),
        )
        h = _silu(h)
        h = self._conv(f"{prefix}.conv2", h, padding=1)

        shortcut_key = f"{prefix}.conv_shortcut.weight"
        if shortcut_key in self.w:
            x = self._conv(f"{prefix}.conv_shortcut", x, padding=0)
        return x + h

    def _attn(self, prefix: str, x: mx.array) -> mx.array:
        h = _group_norm(
            x,
            self._weight(f"{prefix}.group_norm.weight"),
            self._weight(f"{prefix}.group_norm.bias"),
        )
        q = self._conv(f"{prefix}.to_q", h, padding=0)
        k = self._conv(f"{prefix}.to_k", h, padding=0)
        v = self._conv(f"{prefix}.to_v", h, padding=0)

        batch, height, width, channels = q.shape
        q = q.reshape((batch, 1, height * width, channels))
        k = k.reshape((batch, 1, height * width, channels))
        v = v.reshape((batch, 1, height * width, channels))

        out = _attention(q, k, v, scale=1.0 / math.sqrt(channels))
        out = out.reshape((batch, height, width, channels))
        out = self._conv(f"{prefix}.to_out.0", out, padding=0)
        return x + out

    def _up_block(self, idx: int, x: mx.array) -> mx.array:
        prefix = f"decoder.up_blocks.{idx}"
        for j in range(3):
            x = self._resnet(f"{prefix}.resnets.{j}", x)
        if f"{prefix}.upsamplers.0.conv.weight" in self.w:
            x = _upsample_nearest_2x(x)
            x = self._conv(f"{prefix}.upsamplers.0.conv", x, padding=1)
        return x

    def _inv_normalize(self, z: mx.array) -> mx.array:
        running_mean = self._weight("bn.running_mean")
        running_var = self._weight("bn.running_var")
        scale = mx.sqrt(running_var.reshape((1, 1, 1, -1)) + self.bn_eps)
        mean = running_mean.reshape((1, 1, 1, -1))
        return z * scale + mean

    def decode(self, z: mx.array) -> mx.array:
        z = self._inv_normalize(z)
        batch, height, width, channels = z.shape
        z = z.reshape((batch, height, width, 32, self.pixel_shuffle[0], self.pixel_shuffle[1]))
        z = z.transpose((0, 1, 4, 2, 5, 3)).reshape(
            (batch, height * self.pixel_shuffle[0], width * self.pixel_shuffle[1], 32)
        )

        h = self._conv("post_quant_conv", z, padding=0)
        h = self._conv("decoder.conv_in", h, padding=1)

        h = self._resnet("decoder.mid_block.resnets.0", h)
        h = self._attn("decoder.mid_block.attentions.0", h)
        h = self._resnet("decoder.mid_block.resnets.1", h)

        for idx in range(4):
            h = self._up_block(idx, h)

        h = _group_norm(
            h,
            self._weight("decoder.conv_norm_out.weight"),
            self._weight("decoder.conv_norm_out.bias"),
        )
        h = _silu(h)
        return self._conv("decoder.conv_out", h, padding=1)


class Flux2Qwen3TextEncoder(GGUFQwen):
    def __init__(
        self,
        weights: dict[str, mx.array],
        meta: dict[str, Any],
        *,
        hidden_state_indices: tuple[int, int, int] = (9, 18, 27),
    ):
        super().__init__(weights, meta)
        if self.cfg.arch != "qwen3":
            raise ValueError(
                f"Expected a qwen3 GGUF text encoder, got architecture {self.cfg.arch!r}."
            )
        self.hidden_state_indices = hidden_state_indices

    def encode_hidden_states(self, token_ids: mx.array) -> mx.array:
        x = self._embed(token_ids).astype(self._compute_dtype)
        batch, seq_len, _ = x.shape
        hidden_states: list[mx.array] = [x]

        for blk in self._blocks:
            resid = x
            x_norm = self._rms_norm(x, blk.attn_norm)
            q = blk.attn_q(x_norm).reshape((batch, seq_len, self.cfg.n_heads, self.cfg.head_dim))
            k = blk.attn_k(x_norm).reshape((batch, seq_len, self.cfg.n_kv_heads, self.cfg.head_dim))
            v = blk.attn_v(x_norm).reshape((batch, seq_len, self.cfg.n_kv_heads, self.cfg.head_dim))

            if blk.q_norm is not None:
                q = self._rms_norm(q, blk.q_norm)
            if blk.k_norm is not None:
                k = self._rms_norm(k, blk.k_norm)

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
            attn_width = int(ctx.shape[1]) * int(ctx.shape[3])
            ctx = ctx.transpose((0, 2, 1, 3)).reshape((batch, seq_len, attn_width))
            x = resid + blk.attn_output(ctx)

            resid = x
            x_norm = self._rms_norm(x, blk.ffn_norm)
            up = blk.ffn_up(x_norm)
            gate = blk.ffn_gate(x_norm)
            x = resid + blk.ffn_down(_silu(gate) * up)
            hidden_states.append(x)

        selected: list[mx.array] = []
        for idx in self.hidden_state_indices:
            if idx < 0 or idx >= len(hidden_states):
                raise ValueError(
                    f"Requested hidden state index {idx}, but only {len(hidden_states)} are available."
                )
            selected.append(hidden_states[idx].astype(self._compute_dtype))
        return mx.concatenate(selected, axis=-1)


def _parse_hidden_state_indices(raw: str) -> tuple[int, ...]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if not parts:
        raise ValueError("At least one hidden-state index is required.")
    return tuple(int(x) for x in parts)


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


def _sample_image_noise(seed: int, height: int, width: int, channels: int) -> mx.array:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((1, height, width, channels), dtype=np.float32)
    return mx.array(data.astype(np.float16))


def _prepare_prompt_features(
    encoder: Flux2Qwen3TextEncoder,
    tokenizer: Any,
    prompt: str,
    *,
    verbose: bool,
) -> tuple[mx.array, mx.array, list[int]]:
    prompt_text = apply_qwen_chat_template(prompt)
    token_ids = tokenizer.encode(prompt_text)
    if not token_ids:
        raise ValueError("Prompt tokenization returned no tokens.")
    token_arr = mx.array([token_ids], dtype=mx.int32)
    ctx = encoder.encode_hidden_states(token_arr)
    ctx_ids = _build_text_position_ids(1, ctx.shape[1])
    if verbose:
        print(f"[text] tokens={len(token_ids)} ctx_shape={tuple(ctx.shape)}")
    return ctx, ctx_ids, token_ids


def _describe_context_mismatch(
    *,
    expected_context_dim: int,
    encoder_hidden_size: int,
    hidden_state_count: int,
    llm_path: str,
) -> str:
    actual_context_dim = encoder_hidden_size * hidden_state_count
    parts = [
        f"Selected text encoder outputs context dim {actual_context_dim}, but the diffusion model expects {expected_context_dim}."
    ]

    required_state_count, remainder = divmod(expected_context_dim, encoder_hidden_size)
    if remainder == 0:
        parts.append(
            f"{Path(llm_path).name} emits {encoder_hidden_size}-wide hidden states, so set --text-hidden-layers to exactly {required_state_count} indices."
        )
    else:
        expected_state_width, width_remainder = divmod(expected_context_dim, hidden_state_count)
        if width_remainder == 0:
            parts.append(
                f"With your current {hidden_state_count} hidden-state indices, the diffusion model expects each hidden state to have width {expected_state_width}, but {Path(llm_path).name} emits width {encoder_hidden_size}."
            )
            if expected_state_width == 4096 and encoder_hidden_size == 2560:
                parts.append(
                    "For Flux.2 klein-9B this typically means using a Qwen3-8B GGUF text encoder instead of Qwen3-4B."
                )

    return " ".join(parts)


def _generate(
    model: Flux2Model,
    vae: Flux2VaeDecoder,
    encoder: Flux2Qwen3TextEncoder,
    tokenizer: Any,
    *,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    verbose: bool,
) -> mx.array:
    if width % 16 != 0 or height % 16 != 0:
        raise ValueError(f"Width/height must both be divisible by 16, got {width}x{height}.")
    if cfg_scale != 1.0:
        print(
            f"[warn] Flux.2 klein models are guidance-distilled; cfg-scale={cfg_scale} is ignored and 1.0 is expected."
        )

    ctx, ctx_ids, _ = _prepare_prompt_features(encoder, tokenizer, prompt, verbose=verbose)
    latent_h = height // 16
    latent_w = width // 16
    latent = _sample_image_noise(seed, latent_h, latent_w, model.cfg.in_channels)
    img = _flatten_image_tokens(latent)
    img_ids = _build_image_position_ids(1, latent_h, latent_w)
    timesteps = _get_schedule(steps, int(img.shape[1]))

    if verbose:
        print(
            f"[sample] seed={seed} latent_shape={tuple(latent.shape)} "
            f"img_tokens={img.shape[1]} steps={steps}"
        )

    guidance = mx.full((1,), 1.0, dtype=mx.float32)
    for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = mx.full((1,), t_curr, dtype=mx.float32)
        pred = model.forward(
            img,
            img_ids,
            ctx,
            ctx_ids,
            timesteps=t_vec,
            guidance=guidance,
        )
        img = img + (t_prev - t_curr) * pred
        if verbose:
            print(f"[step {step_idx + 1}/{steps}] t={t_curr:.6f}->{t_prev:.6f}")

    decoded = vae.decode(_unflatten_image_tokens(img, latent_h, latent_w))
    return decoded


def _save_image(image: mx.array, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    x = mx.clip(image[0], -1.0, 1.0)
    x = ((x + 1.0) * 127.5).astype(mx.uint8)
    pil = Image.fromarray(np.array(x))
    pil.save(output)


def _build_default_output_path() -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("output") / f"flux2-{stamp}.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Flux.2 klein text-to-image inference in MLX using GGUF diffusion/text weights and a safetensors VAE."
    )
    parser.add_argument(
        "--diffusion-model",
        type=str,
        default="~/Downloads/flux-2-klein-4b-Q4_0.gguf",
        help="Path to the Flux.2 klein GGUF diffusion model, such as the 4B or 9B variant.",
    )
    parser.add_argument(
        "--vae",
        type=str,
        default="~/Downloads/flux2-vae.safetensors",
        help="Path to the Flux.2 VAE safetensors file.",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="~/Downloads/Qwen3-4B-Q4_0.gguf",
        help="Path to the Qwen3 GGUF text encoder. Use Qwen3-4B for Flux.2 klein-4B, or Qwen3-8B for Flux.2 klein-9B.",
    )
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Prompt text.")
    parser.add_argument("--width", type=int, default=1360, help="Output width. Must be divisible by 16.")
    parser.add_argument("--height", type=int, default=768, help="Output height. Must be divisible by 16.")
    parser.add_argument("--steps", type=int, default=4, help="Number of denoising steps.")
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        help="Compatibility flag with sd.cpp. Flux.2 klein models expect 1.0.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path. Defaults to output/flux2-<timestamp>.png.",
    )
    parser.add_argument(
        "--text-hidden-layers",
        type=str,
        default="9,18,27",
        help="Comma-separated hidden-state indices taken from Qwen3 hidden_states.",
    )
    parser.add_argument("--cpu", action="store_true", help="Run on CPU.")
    parser.add_argument(
        "--offload-to-cpu",
        action="store_true",
        help="Alias for --cpu for sd.cpp compatibility.",
    )
    parser.add_argument(
        "--diffusion-fa",
        action="store_true",
        help="Accepted for sd.cpp CLI compatibility. MLX uses fast attention directly.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    if args.cpu or args.offload_to_cpu:
        mx.set_default_device(mx.cpu)

    hidden_state_indices = _parse_hidden_state_indices(args.text_hidden_layers)
    output_path = Path(args.output).expanduser() if args.output else _build_default_output_path()

    t0 = time.perf_counter()
    diff_weights, _ = _load_gguf(args.diffusion_model)
    model_cfg = Flux2Config.from_weights(diff_weights)
    model = Flux2Model(diff_weights, model_cfg)

    llm_weights, llm_meta = _load_gguf(args.llm)
    tokenizer = build_tokenizer(llm_meta)
    if tokenizer is None:
        raise ValueError("Failed to construct a tokenizer from the Qwen3 GGUF metadata.")
    encoder = Flux2Qwen3TextEncoder(
        llm_weights,
        llm_meta,
        hidden_state_indices=hidden_state_indices,
    )
    context_dim = encoder.cfg.d_model * len(hidden_state_indices)
    if context_dim != model_cfg.context_in_dim:
        raise ValueError(
            _describe_context_mismatch(
                expected_context_dim=model_cfg.context_in_dim,
                encoder_hidden_size=encoder.cfg.d_model,
                hidden_state_count=len(hidden_state_indices),
                llm_path=args.llm,
            )
        )

    vae_weights = _load_safetensors(args.vae)
    vae = Flux2VaeDecoder(vae_weights)
    if args.verbose:
        print(
            f"[model] hidden_size={model_cfg.hidden_size} context_in_dim={model_cfg.context_in_dim} "
            f"double_blocks={model_cfg.depth} single_blocks={model_cfg.depth_single_blocks} "
            f"heads={model_cfg.num_heads}"
        )
        print(f"[load] completed in {time.perf_counter() - t0:.3f}s")

    t1 = time.perf_counter()
    image = _generate(
        model,
        vae,
        encoder,
        tokenizer,
        prompt=args.prompt,
        seed=args.seed,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        verbose=args.verbose,
    )
    mx.eval(image)
    if args.verbose:
        print(f"[generate] completed in {time.perf_counter() - t1:.3f}s")

    _save_image(image, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
