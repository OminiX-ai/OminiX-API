#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import struct
import time
from pathlib import Path
from typing import Any, Callable

import imageio.v2 as imageio
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece as spm
from PIL import Image, ImageDraw
from sentencepiece import sentencepiece_model_pb2 as sp_pb2

from infer_qwen_gguf import resolve_affine_quant_params


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
WAN_VAE_MEAN = [
    -0.2289,
    -0.0052,
    -0.1323,
    -0.2339,
    -0.2799,
    0.0174,
    0.1838,
    0.1557,
    -0.1382,
    0.0542,
    0.2813,
    0.0891,
    0.1570,
    -0.0098,
    0.0375,
    -0.1825,
    -0.2246,
    -0.1207,
    -0.0698,
    0.5109,
    0.2665,
    -0.2108,
    -0.2158,
    0.2502,
    -0.2055,
    -0.0322,
    0.1109,
    0.1567,
    -0.0729,
    0.0899,
    -0.2799,
    -0.1230,
    -0.0313,
    -0.1649,
    0.0117,
    0.0723,
    -0.2839,
    -0.2083,
    -0.0520,
    0.3748,
    0.0152,
    0.1957,
    0.1433,
    -0.2944,
    0.3573,
    -0.0548,
    -0.1681,
    -0.0667,
]
WAN_VAE_INV_STD = [
    1.0 / x
    for x in [
        0.4765,
        1.0364,
        0.4514,
        1.1677,
        0.5313,
        0.4990,
        0.4818,
        0.5013,
        0.8158,
        1.0344,
        0.5894,
        1.0901,
        0.6885,
        0.6165,
        0.8454,
        0.4978,
        0.5759,
        0.3523,
        0.7135,
        0.6804,
        0.5833,
        1.4146,
        0.8986,
        0.5659,
        0.7069,
        0.5338,
        0.4889,
        0.4917,
        0.4069,
        0.4999,
        0.6866,
        0.4093,
        0.5709,
        0.6065,
        0.6415,
        0.4944,
        0.5726,
        1.2042,
        0.5458,
        1.6887,
        0.3971,
        1.0600,
        0.3943,
        0.5537,
        0.5444,
        0.4089,
        0.7468,
        0.7744,
    ]
]

LinearFn = Callable[[mx.array], mx.array]

GGUF_VALUE_TYPE_UINT8 = 0
GGUF_VALUE_TYPE_INT8 = 1
GGUF_VALUE_TYPE_UINT16 = 2
GGUF_VALUE_TYPE_INT16 = 3
GGUF_VALUE_TYPE_UINT32 = 4
GGUF_VALUE_TYPE_INT32 = 5
GGUF_VALUE_TYPE_FLOAT32 = 6
GGUF_VALUE_TYPE_BOOL = 7
GGUF_VALUE_TYPE_STRING = 8
GGUF_VALUE_TYPE_ARRAY = 9
GGUF_VALUE_TYPE_UINT64 = 10
GGUF_VALUE_TYPE_INT64 = 11
GGUF_VALUE_TYPE_FLOAT64 = 12

_GGUF_SCALAR_FMTS: dict[int, tuple[str, int]] = {
    GGUF_VALUE_TYPE_UINT8: ("<B", 1),
    GGUF_VALUE_TYPE_INT8: ("<b", 1),
    GGUF_VALUE_TYPE_UINT16: ("<H", 2),
    GGUF_VALUE_TYPE_INT16: ("<h", 2),
    GGUF_VALUE_TYPE_UINT32: ("<I", 4),
    GGUF_VALUE_TYPE_INT32: ("<i", 4),
    GGUF_VALUE_TYPE_FLOAT32: ("<f", 4),
    GGUF_VALUE_TYPE_BOOL: ("<B", 1),
    GGUF_VALUE_TYPE_UINT64: ("<Q", 8),
    GGUF_VALUE_TYPE_INT64: ("<q", 8),
    GGUF_VALUE_TYPE_FLOAT64: ("<d", 8),
}


def _meta_scalar(meta: dict[str, Any], key: str, default: Any = None) -> Any:
    if key not in meta:
        return default
    value = meta[key]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _meta_int(meta: dict[str, Any], key: str, default: int) -> int:
    return int(_meta_scalar(meta, key, default))


def _meta_bool(meta: dict[str, Any], key: str, default: bool) -> bool:
    return bool(_meta_scalar(meta, key, default))


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


def _device_from_name(name: str):
    if name == "cpu":
        return mx.cpu
    if name == "gpu":
        return mx.gpu
    raise ValueError(f"Unsupported device name: {name!r}")


def _dtype_from_name(name: str) -> mx.Dtype:
    if name == "float16":
        return mx.float16
    if name == "bfloat16":
        return mx.bfloat16
    if name == "float32":
        return mx.float32
    raise ValueError(f"Unsupported dtype name: {name!r}")


def _convert_floating_weights_dtype(weights: dict[str, mx.array], dtype: mx.Dtype) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    for key, value in weights.items():
        if hasattr(value, "dtype") and mx.issubdtype(value.dtype, mx.floating) and value.dtype != dtype:
            out[key] = value.astype(dtype)
        else:
            out[key] = value
    return out


def _build_vae_decoder(path: str, device_name: str, dtype_name: str) -> "WanVAE22Decoder":
    previous_device = mx.default_device()
    mx.set_default_device(_device_from_name(device_name))
    try:
        weights = _load_safetensors(path)
        dtype = _dtype_from_name(dtype_name)
        weights = _convert_floating_weights_dtype(weights, dtype)
        return WanVAE22Decoder(weights, compute_dtype=dtype)
    finally:
        mx.set_default_device(previous_device)


def _copy_array_to_device(x: mx.array, device_name: str) -> mx.array:
    previous_device = mx.default_device()
    mx.set_default_device(_device_from_name(device_name))
    try:
        return mx.array(np.array(x))
    finally:
        mx.set_default_device(previous_device)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _gelu_approx(x: mx.array) -> mx.array:
    return nn.gelu_approx(x)


def _layer_norm(
    x: mx.array,
    weight: mx.array | None = None,
    bias: mx.array | None = None,
    *,
    eps: float = 1e-6,
) -> mx.array:
    x32 = x.astype(mx.float32)
    mean = mx.mean(x32, axis=-1, keepdims=True)
    var = mx.mean((x32 - mean) * (x32 - mean), axis=-1, keepdims=True)
    y = (x32 - mean) * mx.rsqrt(var + eps)
    if weight is not None:
        y = y * weight.astype(y.dtype)
    if bias is not None:
        y = y + bias.astype(y.dtype)
    return y.astype(x.dtype)


def _rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    x32 = x.astype(mx.float32)
    rrms = mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    y = x32 * rrms
    return (y * weight.astype(y.dtype)).astype(x.dtype)


def _rms_norm_channel_first(
    x: mx.array,
    gamma: mx.array,
    *,
    images: bool = False,
    eps: float = 1e-12,
) -> mx.array:
    x32 = x.astype(mx.float32)
    norm = mx.sqrt(mx.maximum(mx.sum(x32 * x32, axis=1, keepdims=True), eps))
    y = x32 * (math.sqrt(int(x.shape[1])) / norm)
    gamma = gamma.astype(y.dtype)
    if images:
        gamma = gamma.reshape((1, gamma.shape[0], 1, 1))
    else:
        gamma = gamma.reshape((1, gamma.shape[0], 1, 1, 1))
    return (y * gamma).astype(x.dtype)


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
    y = mx.conv2d(x_hwc, _conv2d_weight(weight), stride=stride, padding=padding)
    if bias is not None:
        y = y + bias.astype(y.dtype)
    return y.transpose((0, 3, 1, 2))


def _conv3d_ncthw(
    x: mx.array,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0),
) -> mx.array:
    x_dhwc = x.transpose((0, 2, 3, 4, 1))
    y = mx.conv3d(x_dhwc, _conv3d_weight(weight), stride=stride, padding=padding)
    if bias is not None:
        y = y + bias.astype(y.dtype)
    return y.transpose((0, 4, 1, 2, 3))


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
    return _conv3d_ncthw(x, weight, bias=bias, stride=stride, padding=(0, 0, 0))


def _causal_conv3d_ncthw_cached(
    x: mx.array,
    weight: mx.array,
    bias: mx.array | None,
    *,
    cache: dict[str, mx.array],
    key: str,
    padding: tuple[int, int, int] = (0, 0, 0),
) -> mx.array:
    pad_t, pad_h, pad_w = padding
    prev = cache.get(key)

    x_in = x
    left_t = 2 * pad_t
    if prev is not None and left_t > 0:
        x_in = mx.concatenate([prev.astype(x.dtype), x], axis=2)
        left_t = max(0, left_t - int(prev.shape[2]))

    if left_t or pad_h or pad_w:
        x_in = mx.pad(
            x_in,
            [
                (0, 0),
                (0, 0),
                (left_t, 0),
                (pad_h, pad_h),
                (pad_w, pad_w),
            ],
        )

    actual = x if prev is None else mx.concatenate([prev, x], axis=2)
    keep = min(2, int(actual.shape[2]))
    cache[key] = actual[:, :, -keep:, :, :]

    return _conv3d_ncthw(x_in, weight, bias=bias, stride=(1, 1, 1), padding=(0, 0, 0))


def _upsample_spatial_2x_ncthw(x: mx.array) -> mx.array:
    batch, channels, time_dim, height, width = x.shape
    x = x.reshape((batch, channels, time_dim, height, 1, width, 1))
    x = mx.broadcast_to(x, (batch, channels, time_dim, height, 2, width, 2))
    return x.reshape((batch, channels, time_dim, height * 2, width * 2))


def _flatten_video_frames(x: mx.array) -> tuple[mx.array, int, int]:
    batch, channels, time_dim, height, width = x.shape
    y = x.transpose((0, 2, 1, 3, 4)).reshape((batch * time_dim, channels, height, width))
    return y, batch, time_dim


def _restore_video_frames(x: mx.array, batch: int, time_dim: int) -> mx.array:
    _, channels, height, width = x.shape
    return x.reshape((batch, time_dim, channels, height, width)).transpose((0, 2, 1, 3, 4))


def _timestep_embedding(dim: int, timesteps: mx.array) -> mx.array:
    if dim % 2 != 0:
        raise ValueError(f"Sinusoidal embedding dim must be even, got {dim}.")
    half = dim // 2
    positions = timesteps.astype(mx.float32).reshape((-1,))
    freqs = mx.exp(
        -math.log(10000.0) * mx.arange(0, half, dtype=mx.float32) / max(half, 1)
    )
    angles = positions[:, None] * freqs[None, :]
    return mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1)


def _merge_heads(x: mx.array) -> mx.array:
    batch, heads, seq_len, head_dim = x.shape
    return x.transpose((0, 2, 1, 3)).reshape((batch, seq_len, heads * head_dim))


def _count_prefix_indices(weights: dict[str, mx.array], prefix: str) -> int:
    indices: set[int] = set()
    plen = len(prefix) + 1
    for key in weights:
        if not key.startswith(prefix + "."):
            continue
        rest = key[plen:]
        num, _, _ = rest.partition(".")
        if num.isdigit():
            indices.add(int(num))
    if not indices:
        raise ValueError(f"Cannot find indexed weights under {prefix!r}.")
    expected = set(range(max(indices) + 1))
    if indices != expected:
        raise ValueError(
            f"Non-contiguous indices for {prefix!r}: expected {sorted(expected)}, got {sorted(indices)}."
        )
    return len(indices)


def _read_gguf_string(handle) -> str:
    (length,) = struct.unpack("<Q", handle.read(8))
    return handle.read(length).decode("utf-8", errors="replace")


def _read_gguf_scalar(handle, value_type: int) -> Any:
    fmt, size = _GGUF_SCALAR_FMTS[value_type]
    (value,) = struct.unpack(fmt, handle.read(size))
    if value_type == GGUF_VALUE_TYPE_BOOL:
        return bool(value)
    return value


def _read_gguf_value(handle, value_type: int) -> Any:
    if value_type == GGUF_VALUE_TYPE_STRING:
        return _read_gguf_string(handle)
    if value_type == GGUF_VALUE_TYPE_ARRAY:
        elem_type, length = struct.unpack("<IQ", handle.read(12))
        if elem_type == GGUF_VALUE_TYPE_STRING:
            return [_read_gguf_string(handle) for _ in range(length)]
        fmt, size = _GGUF_SCALAR_FMTS[elem_type]
        raw = handle.read(size * length)
        values = list(struct.unpack(f"<{length}{fmt[-1]}", raw))
        if elem_type == GGUF_VALUE_TYPE_BOOL:
            return [bool(v) for v in values]
        return values
    return _read_gguf_scalar(handle, value_type)


def _load_gguf_metadata_raw(path: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    with Path(path).expanduser().open("rb") as handle:
        magic, version, tensor_count, metadata_count = struct.unpack("<IIQQ", handle.read(24))
        if magic != 0x46554747:
            raise ValueError(f"{path} is not a GGUF file.")
        if version != 3:
            raise ValueError(f"Unsupported GGUF version {version} in {path}.")
        del tensor_count
        for _ in range(metadata_count):
            key = _read_gguf_string(handle)
            (value_type,) = struct.unpack("<I", handle.read(4))
            meta[key] = _read_gguf_value(handle, value_type)
    return meta


class GGUFSentencePieceTokenizer:
    def __init__(
        self,
        processor: spm.SentencePieceProcessor,
        *,
        add_bos_token: bool,
        add_eos_token: bool,
        bos_id: int | None,
        eos_id: int | None,
    ):
        self.processor = processor
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.bos_id = bos_id
        self.eos_id = eos_id

    @classmethod
    def from_gguf(cls, path: str) -> "GGUFSentencePieceTokenizer":
        meta = _load_gguf_metadata_raw(path)
        tokens = meta.get("tokenizer.ggml.tokens")
        scores = meta.get("tokenizer.ggml.scores")
        token_types = meta.get("tokenizer.ggml.token_type")
        if not isinstance(tokens, list) or not isinstance(scores, list) or not isinstance(token_types, list):
            raise ValueError("GGUF tokenizer metadata is incomplete for sentencepiece reconstruction.")
        if not (len(tokens) == len(scores) == len(token_types)):
            raise ValueError("GGUF tokenizer metadata has mismatched tokens/scores/token_type lengths.")

        bos_id = meta.get("tokenizer.ggml.bos_token_id")
        eos_id = meta.get("tokenizer.ggml.eos_token_id")
        pad_id = meta.get("tokenizer.ggml.padding_token_id")
        unk_id = meta.get("tokenizer.ggml.unknown_token_id")

        token_to_id = {token: idx for idx, token in enumerate(tokens)}
        if bos_id is None:
            bos_id = token_to_id.get("<s>")
        if eos_id is None:
            eos_id = token_to_id.get("</s>")
        if pad_id is None:
            pad_id = token_to_id.get("<pad>")
        if unk_id is None:
            unk_id = token_to_id.get("<unk>", 0)

        proto = sp_pb2.ModelProto()
        proto.trainer_spec.model_type = sp_pb2.TrainerSpec.UNIGRAM
        proto.trainer_spec.vocab_size = len(tokens)
        proto.trainer_spec.unk_id = int(unk_id)
        proto.trainer_spec.pad_id = int(pad_id) if pad_id is not None else -1
        proto.trainer_spec.eos_id = int(eos_id) if eos_id is not None else -1
        proto.trainer_spec.bos_id = int(bos_id) if bos_id is not None else -1
        proto.trainer_spec.byte_fallback = True
        proto.normalizer_spec.add_dummy_prefix = bool(meta.get("tokenizer.ggml.add_space_prefix", True))
        proto.normalizer_spec.escape_whitespaces = True
        proto.normalizer_spec.remove_extra_whitespaces = bool(
            meta.get("tokenizer.ggml.remove_extra_whitespaces", False)
        )

        piece_type_map = {
            0: sp_pb2.ModelProto.SentencePiece.NORMAL,
            1: sp_pb2.ModelProto.SentencePiece.NORMAL,
            2: sp_pb2.ModelProto.SentencePiece.UNKNOWN,
            3: sp_pb2.ModelProto.SentencePiece.CONTROL,
            4: sp_pb2.ModelProto.SentencePiece.USER_DEFINED,
            5: sp_pb2.ModelProto.SentencePiece.UNUSED,
            6: sp_pb2.ModelProto.SentencePiece.BYTE,
        }

        for token, score, token_type in zip(tokens, scores, token_types):
            piece = proto.pieces.add()
            piece.piece = str(token)
            piece.score = float(score)
            piece.type = piece_type_map.get(int(token_type), sp_pb2.ModelProto.SentencePiece.NORMAL)

        processor = spm.SentencePieceProcessor()
        processor.LoadFromSerializedProto(proto.SerializeToString())
        return cls(
            processor,
            add_bos_token=bool(meta.get("tokenizer.ggml.add_bos_token", False)),
            add_eos_token=bool(meta.get("tokenizer.ggml.add_eos_token", True)),
            bos_id=int(bos_id) if bos_id is not None else None,
            eos_id=int(eos_id) if eos_id is not None else None,
        )

    def encode(self, text: str, *, max_length: int) -> list[int]:
        ids = [int(x) for x in self.processor.EncodeAsIds(text)]
        if self.add_bos_token and self.bos_id is not None:
            ids = [self.bos_id] + ids
        if self.add_eos_token and self.eos_id is not None:
            ids = ids + [self.eos_id]
        if len(ids) > max_length:
            ids = ids[:max_length]
            if self.add_eos_token and self.eos_id is not None and max_length > 0:
                ids[-1] = self.eos_id
        return ids


class GGUFModule:
    def __init__(self, weights: dict[str, mx.array], meta: dict[str, Any], *, compute_dtype: mx.Dtype | None = None):
        self.w = weights
        self.meta = meta
        self.compute_dtype = compute_dtype if compute_dtype is not None else _infer_compute_dtype(weights)
        self._quant_params_cache: dict[str, Any] = {}

    def _has_quantized_triplet(self, prefix: str) -> bool:
        return (
            f"{prefix}.weight" in self.w
            and f"{prefix}.scales" in self.w
            and f"{prefix}.biases" in self.w
        )

    def _get_quant_params(self, prefix: str):
        if prefix not in self._quant_params_cache:
            self._quant_params_cache[prefix] = resolve_affine_quant_params(self.w, self.meta, prefix)
        return self._quant_params_cache[prefix]

    def _linear(self, prefix: str, x: mx.array) -> mx.array:
        bias = self.w.get(f"{prefix}.bias")
        if self._has_quantized_triplet(prefix):
            params = self._get_quant_params(prefix)
            y = mx.quantized_matmul(
                x,
                self.w[f"{prefix}.weight"],
                self.w[f"{prefix}.scales"],
                self.w[f"{prefix}.biases"],
                transpose=True,
                group_size=params.group_size,
                bits=params.bits,
                mode=params.mode,
            )
        else:
            y = x @ self.w[f"{prefix}.weight"].T
        if bias is not None:
            y = y + bias.astype(y.dtype)
        return y

    def _embedding_lookup(self, prefix: str, ids: np.ndarray, vocab_size: int) -> mx.array:
        if not self._has_quantized_triplet(prefix):
            return self.w[f"{prefix}.weight"][mx.array(ids, dtype=mx.int32)]

        params = self._get_quant_params(prefix)
        one_hot = np.zeros((len(ids), vocab_size), dtype=np.float16)
        one_hot[np.arange(len(ids)), ids] = 1.0
        return mx.quantized_matmul(
            mx.array(one_hot, dtype=self.compute_dtype),
            self.w[f"{prefix}.weight"],
            self.w[f"{prefix}.scales"],
            self.w[f"{prefix}.biases"],
            transpose=False,
            group_size=params.group_size,
            bits=params.bits,
            mode=params.mode,
        )

    def _dequantize_rows(self, prefix: str, row_ids: mx.array) -> mx.array:
        params = self._get_quant_params(prefix)
        weight = self.w[f"{prefix}.weight"][row_ids]
        scales = self.w[f"{prefix}.scales"][row_ids]
        biases = self.w[f"{prefix}.biases"][row_ids]
        return mx.dequantize(
            weight,
            scales,
            biases,
            group_size=params.group_size,
            bits=params.bits,
            mode=params.mode,
            dtype=self.compute_dtype,
        )


class UMT5Encoder(GGUFModule):
    def __init__(self, weights: dict[str, mx.array], meta: dict[str, Any]):
        super().__init__(weights, meta, compute_dtype=mx.bfloat16)
        arch = str(_meta_scalar(meta, "general.architecture", "t5encoder"))
        if arch != "t5encoder":
            raise ValueError(f"Expected a t5encoder GGUF, got {arch!r}.")
        self.dim = _meta_int(meta, "t5encoder.embedding_length", 4096)
        self.dim_ffn = _meta_int(meta, "t5encoder.feed_forward_length", 10240)
        self.num_heads = _meta_int(meta, "t5encoder.attention.head_count", 64)
        self.key_length = _meta_int(meta, "t5encoder.attention.key_length", 64)
        self.value_length = _meta_int(meta, "t5encoder.attention.value_length", 64)
        self.num_layers = _meta_int(meta, "t5encoder.block_count", 24)
        self.num_buckets = _meta_int(meta, "t5encoder.attention.relative_buckets_count", 32)
        self.eps = float(_meta_scalar(meta, "t5encoder.attention.layer_norm_epsilon", 1e-6))
        self.text_len = _meta_int(meta, "t5encoder.context_length", 512)
        self.vocab_size = len(meta.get("tokenizer.ggml.tokens", []))
        self._rel_bucket_cache: dict[int, np.ndarray] = {}

    def _embed(self, token_ids: mx.array) -> mx.array:
        prefix = "token_embd"
        if self._has_quantized_triplet(prefix):
            ids = np.array(token_ids, dtype=np.int32).reshape((-1,))
            return self._embedding_lookup(prefix, ids, self.vocab_size)
        return self.w[f"{prefix}.weight"][token_ids]

    def _relative_position_bucket(self, seq_len: int) -> np.ndarray:
        if seq_len in self._rel_bucket_cache:
            return self._rel_bucket_cache[seq_len]

        context_pos = np.arange(seq_len, dtype=np.int32)[:, None]
        memory_pos = np.arange(seq_len, dtype=np.int32)[None, :]
        rel_pos = memory_pos - context_pos

        num_buckets = self.num_buckets // 2
        rel_buckets = (rel_pos > 0).astype(np.int32) * num_buckets
        rel_pos = np.abs(rel_pos).astype(np.int32)

        max_exact = num_buckets // 2
        rel_pos_clamped = np.maximum(rel_pos, 1)
        rel_pos_large = max_exact + (
            np.log(rel_pos_clamped / max_exact) / math.log(128.0 / max_exact) * (num_buckets - max_exact)
        ).astype(np.int32)
        rel_pos_large = np.minimum(rel_pos_large, num_buckets - 1)
        rel_buckets += np.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        self._rel_bucket_cache[seq_len] = rel_buckets
        return rel_buckets

    def _relative_bias(self, layer_idx: int, seq_len: int) -> mx.array:
        buckets = self._relative_position_bucket(seq_len)
        weight = np.array(self.w[f"enc.blk.{layer_idx}.attn_rel_b.weight"], dtype=np.float32)
        bias = weight[buckets]
        return mx.array(np.transpose(bias, (2, 0, 1))[None, ...])

    def _attention(self, layer_idx: int, x: mx.array) -> mx.array:
        batch, seq_len, _ = x.shape
        q = self._linear(f"enc.blk.{layer_idx}.attn_q", x)
        k = self._linear(f"enc.blk.{layer_idx}.attn_k", x)
        v = self._linear(f"enc.blk.{layer_idx}.attn_v", x)

        q = q.reshape((batch, seq_len, self.num_heads, self.key_length)).transpose((0, 2, 1, 3))
        k = k.reshape((batch, seq_len, self.num_heads, self.key_length)).transpose((0, 2, 1, 3))
        v = v.reshape((batch, seq_len, self.num_heads, self.value_length)).transpose((0, 2, 1, 3))

        scores = (q.astype(mx.float32) @ k.astype(mx.float32).transpose((0, 1, 3, 2)))
        scores = scores + self._relative_bias(layer_idx, seq_len).astype(scores.dtype)
        attn = mx.softmax(scores, axis=-1).astype(v.dtype)
        ctx = attn @ v
        return self._linear(f"enc.blk.{layer_idx}.attn_o", _merge_heads(ctx))

    def _ffn(self, layer_idx: int, x: mx.array) -> mx.array:
        up = self._linear(f"enc.blk.{layer_idx}.ffn_up", x)
        gate = self._linear(f"enc.blk.{layer_idx}.ffn_gate", x)
        return self._linear(f"enc.blk.{layer_idx}.ffn_down", up * _gelu_approx(gate))

    def encode(self, token_ids: list[int]) -> mx.array:
        ids = mx.array(token_ids, dtype=mx.int32)
        x = self._embed(ids).reshape((1, len(token_ids), self.dim)).astype(self.compute_dtype)
        for layer_idx in range(self.num_layers):
            x = x + self._attention(layer_idx, _rms_norm(x, self.w[f"enc.blk.{layer_idx}.attn_norm.weight"], self.eps))
            x = x + self._ffn(layer_idx, _rms_norm(x, self.w[f"enc.blk.{layer_idx}.ffn_norm.weight"], self.eps))
        x = _rms_norm(x, self.w["enc.output_norm.weight"], self.eps)
        return x[0]


class WanTransformer(GGUFModule):
    def __init__(self, weights: dict[str, mx.array], meta: dict[str, Any]):
        super().__init__(weights, meta, compute_dtype=mx.bfloat16)
        arch = str(_meta_scalar(meta, "general.architecture", "wan"))
        if arch != "wan":
            raise ValueError(f"Expected a Wan GGUF, got {arch!r}.")
        patch_weight = self.w["patch_embedding.weight"]
        self.in_dim = int(patch_weight.shape[1])
        self.dim = int(patch_weight.shape[0])
        self.patch_size = tuple(int(x) for x in patch_weight.shape[2:])
        self.num_heads = _meta_int(meta, "wan.attention.head_count", 24)
        self.num_layers = _count_prefix_indices(weights, "blocks")
        self.head_dim = self.dim // self.num_heads
        self.freq_dim = int(self.w["time_embedding.0.weight"].shape[1])
        self.out_dim = int(self.w["head.head.weight"].shape[0]) // math.prod(self.patch_size)
        self.text_dim = int(self.w["text_embedding.0.weight"].shape[1])
        self.text_len = _meta_int(meta, "wan.context_length", 512)
        self.attn_scale = 1.0 / math.sqrt(self.head_dim)
        self._rope_cache: dict[tuple[int, int, int], tuple[mx.array, mx.array]] = {}

    def _build_rope(self, grid_t: int, grid_h: int, grid_w: int) -> tuple[mx.array, mx.array]:
        key = (grid_t, grid_h, grid_w)
        if key in self._rope_cache:
            return self._rope_cache[key]

        dims = [
            self.head_dim - 4 * (self.head_dim // 6),
            2 * (self.head_dim // 6),
            2 * (self.head_dim // 6),
        ]

        def axis_freq(size: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
            half = dim // 2
            omega = 1.0 / np.power(10000.0, np.arange(0, dim, 2, dtype=np.float32) / dim)
            angles = np.arange(size, dtype=np.float32)[:, None] * omega[None, :]
            return np.cos(angles), np.sin(angles)

        t_cos, t_sin = axis_freq(grid_t, dims[0])
        h_cos, h_sin = axis_freq(grid_h, dims[1])
        w_cos, w_sin = axis_freq(grid_w, dims[2])

        cos = np.concatenate(
            [
                np.broadcast_to(t_cos[:, None, None, :], (grid_t, grid_h, grid_w, t_cos.shape[-1])),
                np.broadcast_to(h_cos[None, :, None, :], (grid_t, grid_h, grid_w, h_cos.shape[-1])),
                np.broadcast_to(w_cos[None, None, :, :], (grid_t, grid_h, grid_w, w_cos.shape[-1])),
            ],
            axis=-1,
        ).reshape((grid_t * grid_h * grid_w, self.head_dim // 2))
        sin = np.concatenate(
            [
                np.broadcast_to(t_sin[:, None, None, :], (grid_t, grid_h, grid_w, t_sin.shape[-1])),
                np.broadcast_to(h_sin[None, :, None, :], (grid_t, grid_h, grid_w, h_sin.shape[-1])),
                np.broadcast_to(w_sin[None, None, :, :], (grid_t, grid_h, grid_w, w_sin.shape[-1])),
            ],
            axis=-1,
        ).reshape((grid_t * grid_h * grid_w, self.head_dim // 2))
        value = (mx.array(cos), mx.array(sin))
        self._rope_cache[key] = value
        return value

    def _apply_rope(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x_ = x.astype(mx.float32).reshape((x.shape[0], x.shape[1], x.shape[2], x.shape[3] // 2, 2))
        real = x_[..., 0]
        imag = x_[..., 1]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        out = mx.stack([real * cos - imag * sin, real * sin + imag * cos], axis=-1)
        return out.reshape(x.shape).astype(x.dtype)

    def _self_attention(self, layer_idx: int, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        q = self._linear(f"blocks.{layer_idx}.self_attn.q", x)
        k = self._linear(f"blocks.{layer_idx}.self_attn.k", x)
        v = self._linear(f"blocks.{layer_idx}.self_attn.v", x)

        q = _rms_norm(q, self.w[f"blocks.{layer_idx}.self_attn.norm_q.weight"])
        k = _rms_norm(k, self.w[f"blocks.{layer_idx}.self_attn.norm_k.weight"])

        q = q.reshape((x.shape[0], x.shape[1], self.num_heads, self.head_dim))
        k = k.reshape((x.shape[0], x.shape[1], self.num_heads, self.head_dim))
        v = v.reshape((x.shape[0], x.shape[1], self.num_heads, self.head_dim))

        q = self._apply_rope(q, cos, sin).transpose((0, 2, 1, 3))
        k = self._apply_rope(k, cos, sin).transpose((0, 2, 1, 3))
        v = v.transpose((0, 2, 1, 3))

        ctx = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.attn_scale)
        return self._linear(f"blocks.{layer_idx}.self_attn.o", _merge_heads(ctx))

    def _cross_attention(self, layer_idx: int, x: mx.array, context: mx.array) -> mx.array:
        q = self._linear(f"blocks.{layer_idx}.cross_attn.q", x)
        k = self._linear(f"blocks.{layer_idx}.cross_attn.k", context)
        v = self._linear(f"blocks.{layer_idx}.cross_attn.v", context)

        q = _rms_norm(q, self.w[f"blocks.{layer_idx}.cross_attn.norm_q.weight"])
        k = _rms_norm(k, self.w[f"blocks.{layer_idx}.cross_attn.norm_k.weight"])

        q = q.reshape((x.shape[0], x.shape[1], self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        k = k.reshape((context.shape[0], context.shape[1], self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        v = v.reshape((context.shape[0], context.shape[1], self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))

        ctx = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.attn_scale)
        return self._linear(f"blocks.{layer_idx}.cross_attn.o", _merge_heads(ctx))

    def _ffn(self, layer_idx: int, x: mx.array) -> mx.array:
        hidden = self._linear(f"blocks.{layer_idx}.ffn.0", x)
        hidden = _gelu_approx(hidden)
        return self._linear(f"blocks.{layer_idx}.ffn.2", hidden)

    def _unpatchify(self, x: mx.array, grid_t: int, grid_h: int, grid_w: int) -> mx.array:
        batch = int(x.shape[0])
        x = x.reshape((batch, grid_t, grid_h, grid_w, *self.patch_size, self.out_dim))
        x = x.transpose((0, 7, 1, 4, 2, 5, 3, 6))
        return x.reshape(
            (
                batch,
                self.out_dim,
                grid_t * self.patch_size[0],
                grid_h * self.patch_size[1],
                grid_w * self.patch_size[2],
            )
        )

    def forward(self, latents: mx.array, sigma: float, context: mx.array) -> mx.array:
        if latents.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported.")
        x = _conv3d_ncthw(
            latents,
            self.w["patch_embedding.weight"],
            self.w.get("patch_embedding.bias"),
            stride=self.patch_size,
            padding=(0, 0, 0),
        )
        _, _, grid_t, grid_h, grid_w = x.shape
        x = x.flatten(2).transpose((0, 2, 1)).astype(self.compute_dtype)

        seq_len = int(x.shape[1])
        cos, sin = self._build_rope(grid_t, grid_h, grid_w)

        if int(context.shape[0]) > self.text_len:
            context = context[: self.text_len]
        elif int(context.shape[0]) < self.text_len:
            pad = mx.zeros((self.text_len - int(context.shape[0]), int(context.shape[1])), dtype=context.dtype)
            context = mx.concatenate([context, pad], axis=0)

        context = context.reshape((1, self.text_len, context.shape[1]))
        context = self._linear("text_embedding.0", context.astype(self.compute_dtype))
        context = _gelu_approx(context)
        context = self._linear("text_embedding.2", context)

        timesteps = mx.full((seq_len,), sigma * 1000.0, dtype=mx.float32)
        time_hidden = _timestep_embedding(self.freq_dim, timesteps).reshape((1, seq_len, self.freq_dim))
        time_hidden = self._linear("time_embedding.0", time_hidden.astype(self.compute_dtype))
        time_hidden = _silu(time_hidden)
        time_hidden = self._linear("time_embedding.2", time_hidden)

        e0 = self._linear("time_projection.1", _silu(time_hidden)).reshape((1, seq_len, 6, self.dim)).astype(mx.float32)

        for layer_idx in range(self.num_layers):
            mod = self.w[f"blocks.{layer_idx}.modulation"].astype(mx.float32).reshape((1, 1, 6, self.dim))
            mods = mod + e0
            shift1, scale1, gate1, shift2, scale2, gate2 = mx.split(mods, 6, axis=2)

            h = _layer_norm(x, eps=1e-6).astype(mx.float32)
            h = h * (1.0 + scale1.squeeze(2)) + shift1.squeeze(2)
            h = self._self_attention(layer_idx, h.astype(self.compute_dtype), cos, sin)
            x = x + h * gate1.squeeze(2).astype(h.dtype)

            cross_in = _layer_norm(
                x,
                self.w[f"blocks.{layer_idx}.norm3.weight"],
                self.w[f"blocks.{layer_idx}.norm3.bias"],
                eps=1e-6,
            )
            x = x + self._cross_attention(layer_idx, cross_in.astype(self.compute_dtype), context)

            h = _layer_norm(x, eps=1e-6).astype(mx.float32)
            h = h * (1.0 + scale2.squeeze(2)) + shift2.squeeze(2)
            h = self._ffn(layer_idx, h.astype(self.compute_dtype))
            x = x + h * gate2.squeeze(2).astype(h.dtype)

        head_mod = self.w["head.modulation"].astype(mx.float32).reshape((1, 1, 2, self.dim))
        head_shift, head_scale = mx.split(head_mod + time_hidden.astype(mx.float32)[:, :, None, :], 2, axis=2)
        y = _layer_norm(x, eps=1e-6).astype(mx.float32)
        y = y * (1.0 + head_scale.squeeze(2)) + head_shift.squeeze(2)
        y = self._linear("head.head", y.astype(self.compute_dtype))
        return self._unpatchify(y, grid_t, grid_h, grid_w)


def _attention_2d_single_head(x: mx.array, prefix: str, weights: dict[str, mx.array]) -> mx.array:
    identity = x
    bt, channels, height, width = x.shape
    x = _rms_norm_channel_first(x, weights[f"{prefix}.norm.gamma"], images=True)
    qkv = _conv2d_nchw(x, weights[f"{prefix}.to_qkv.weight"], weights.get(f"{prefix}.to_qkv.bias"))
    q, k, v = mx.split(qkv, 3, axis=1)
    q = q.reshape((bt, 1, channels, height * width)).transpose((0, 1, 3, 2))
    k = k.reshape((bt, 1, channels, height * width)).transpose((0, 1, 3, 2))
    v = v.reshape((bt, 1, channels, height * width)).transpose((0, 1, 3, 2))
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(channels))
    out = out.transpose((0, 1, 3, 2)).reshape((bt, channels, height, width))
    out = _conv2d_nchw(out, weights[f"{prefix}.proj.weight"], weights.get(f"{prefix}.proj.bias"))
    return out + identity


def _residual_block_3d(x: mx.array, prefix: str, weights: dict[str, mx.array]) -> mx.array:
    residual = x
    h = _rms_norm_channel_first(x, weights[f"{prefix}.residual.0.gamma"], images=False)
    h = _silu(h)
    h = _causal_conv3d_ncthw(h, weights[f"{prefix}.residual.2.weight"], weights.get(f"{prefix}.residual.2.bias"), padding=(1, 1, 1))
    h = _rms_norm_channel_first(h, weights[f"{prefix}.residual.3.gamma"], images=False)
    h = _silu(h)
    h = _causal_conv3d_ncthw(h, weights[f"{prefix}.residual.6.weight"], weights.get(f"{prefix}.residual.6.bias"), padding=(1, 1, 1))
    if f"{prefix}.shortcut.weight" in weights:
        residual = _causal_conv3d_ncthw(
            residual,
            weights[f"{prefix}.shortcut.weight"],
            weights.get(f"{prefix}.shortcut.bias"),
            padding=(0, 0, 0),
        )
    return h + residual


def _dup_up_3d(
    x: mx.array,
    out_channels: int,
    *,
    temporal_factor: int,
    spatial_factor: int,
    first_chunk: bool = False,
) -> mx.array:
    repeats = out_channels * temporal_factor * spatial_factor * spatial_factor // int(x.shape[1])
    x = mx.repeat(x, repeats, axis=1)
    x = x.reshape(
        (
            x.shape[0],
            out_channels,
            temporal_factor,
            spatial_factor,
            spatial_factor,
            x.shape[2],
            x.shape[3],
            x.shape[4],
        )
    )
    x = x.transpose((0, 1, 5, 2, 6, 3, 7, 4))
    x = x.reshape(
        (
            x.shape[0],
            out_channels,
            x.shape[2] * temporal_factor,
            x.shape[4] * spatial_factor,
            x.shape[6] * spatial_factor,
        )
    )
    if first_chunk and temporal_factor > 1:
        x = x[:, :, temporal_factor - 1 :, :, :]
    return x


def _resample_decoder(x: mx.array, prefix: str, weights: dict[str, mx.array], *, temporal: bool) -> mx.array:
    if temporal:
        x = _causal_conv3d_ncthw(x, weights[f"{prefix}.time_conv.weight"], weights.get(f"{prefix}.time_conv.bias"), padding=(1, 0, 0))
        batch, channels2, time_dim, height, width = x.shape
        channels = channels2 // 2
        x = x.reshape((batch, 2, channels, time_dim, height, width))
        x = mx.stack((x[:, 0], x[:, 1]), axis=3).reshape((batch, channels, time_dim * 2, height, width))
        x = x[:, :, 1:, :, :]

    x2d, batch, time_dim = _flatten_video_frames(x)
    x2d = _upsample_spatial_2x_ncthw(_restore_video_frames(x2d, batch, time_dim))
    x2d, batch, time_dim = _flatten_video_frames(x2d)
    x2d = _conv2d_nchw(x2d, weights[f"{prefix}.resample.1.weight"], weights.get(f"{prefix}.resample.1.bias"), padding=(1, 1))
    return _restore_video_frames(x2d, batch, time_dim)


def _unpatchify_rgb_video(x: mx.array, patch_size: int = 2) -> mx.array:
    batch, channels, time_dim, height, width = x.shape
    out_channels = channels // (patch_size * patch_size)
    x = x.reshape((batch, out_channels, patch_size, patch_size, time_dim, height, width))
    x = x.transpose((0, 1, 4, 5, 3, 6, 2))
    return x.reshape((batch, out_channels, time_dim, height * patch_size, width * patch_size))


class WanVAE22Decoder:
    def __init__(self, weights: dict[str, mx.array], *, compute_dtype: mx.Dtype | None = None):
        self.compute_dtype = compute_dtype if compute_dtype is not None else _infer_compute_dtype(weights)
        self.w = _convert_floating_weights_dtype(weights, self.compute_dtype)
        self.mean = mx.array(WAN_VAE_MEAN, dtype=self.compute_dtype).reshape((1, 48, 1, 1, 1))
        self.inv_std = mx.array(WAN_VAE_INV_STD, dtype=self.compute_dtype).reshape((1, 48, 1, 1, 1))
        self.temporal_upsample = [True, True, False]

    def _conv_cached(
        self,
        x: mx.array,
        key: str,
        *,
        padding: tuple[int, int, int] = (0, 0, 0),
        cache: dict[str, mx.array],
    ) -> mx.array:
        return _causal_conv3d_ncthw_cached(
            x,
            self.w[f"{key}.weight"],
            self.w.get(f"{key}.bias"),
            cache=cache,
            key=key,
            padding=padding,
        )

    def _residual_block_chunk(
        self,
        x: mx.array,
        prefix: str,
        *,
        cache: dict[str, mx.array],
    ) -> mx.array:
        residual = x
        h = _rms_norm_channel_first(x, self.w[f"{prefix}.residual.0.gamma"], images=False)
        h = _silu(h)
        h = self._conv_cached(h, f"{prefix}.residual.2", padding=(1, 1, 1), cache=cache)
        h = _rms_norm_channel_first(h, self.w[f"{prefix}.residual.3.gamma"], images=False)
        h = _silu(h)
        h = self._conv_cached(h, f"{prefix}.residual.6", padding=(1, 1, 1), cache=cache)
        if f"{prefix}.shortcut.weight" in self.w:
            residual = _causal_conv3d_ncthw(
                residual,
                self.w[f"{prefix}.shortcut.weight"],
                self.w.get(f"{prefix}.shortcut.bias"),
            )
        return h + residual

    def _resample_chunk(
        self,
        x: mx.array,
        prefix: str,
        *,
        temporal: bool,
        first_chunk: bool,
        cache: dict[str, mx.array],
        warm: dict[str, bool],
    ) -> mx.array:
        if temporal:
            warm_key = f"{prefix}.time_conv.warm"
            if warm.get(warm_key, False):
                x = self._conv_cached(x, f"{prefix}.time_conv", padding=(1, 0, 0), cache=cache)
                batch, channels2, time_dim, height, width = x.shape
                channels = channels2 // 2
                x = x.reshape((batch, 2, channels, time_dim, height, width))
                x = mx.stack((x[:, 0], x[:, 1]), axis=3).reshape((batch, channels, time_dim * 2, height, width))
            else:
                warm[warm_key] = True

        x = _upsample_spatial_2x_ncthw(x)
        x2d, batch, time_dim = _flatten_video_frames(x)
        x2d = _conv2d_nchw(
            x2d,
            self.w[f"{prefix}.resample.1.weight"],
            self.w.get(f"{prefix}.resample.1.bias"),
            padding=(1, 1),
        )
        del first_chunk
        return _restore_video_frames(x2d, batch, time_dim)

    def _decode_chunk(
        self,
        x: mx.array,
        *,
        first_chunk: bool,
        cache: dict[str, mx.array],
        warm: dict[str, bool],
    ) -> mx.array:
        x = self._conv_cached(x, "decoder.conv1", padding=(1, 1, 1), cache=cache)
        x = self._residual_block_chunk(x, "decoder.middle.0", cache=cache)
        x2d, batch, time_dim = _flatten_video_frames(x)
        x2d = _attention_2d_single_head(x2d, "decoder.middle.1", self.w)
        x = _restore_video_frames(x2d, batch, time_dim)
        x = self._residual_block_chunk(x, "decoder.middle.2", cache=cache)

        dims = [1024, 1024, 1024, 512, 256]
        for stage_idx in range(4):
            stage_prefix = f"decoder.upsamples.{stage_idx}.upsamples"
            stage_input = x
            for block_idx in range(3):
                x = self._residual_block_chunk(x, f"{stage_prefix}.{block_idx}", cache=cache)
            if stage_idx != 3:
                temporal = stage_idx < len(self.temporal_upsample) and self.temporal_upsample[stage_idx]
                shortcut = _dup_up_3d(
                    stage_input,
                    out_channels=dims[stage_idx + 1],
                    temporal_factor=2 if temporal else 1,
                    spatial_factor=2,
                    first_chunk=first_chunk,
                )
                main = self._resample_chunk(
                    x,
                    f"{stage_prefix}.3",
                    temporal=temporal,
                    first_chunk=first_chunk,
                    cache=cache,
                    warm=warm,
                )
                x = main + shortcut

        x = _rms_norm_channel_first(x, self.w["decoder.head.0.gamma"], images=False)
        x = _silu(x)
        x = self._conv_cached(x, "decoder.head.2", padding=(1, 1, 1), cache=cache)
        return _unpatchify_rgb_video(x, patch_size=2)

    def decode(self, latents: mx.array, *, verbose: bool = False) -> np.ndarray:
        x = latents.astype(self.compute_dtype)
        x = x / self.inv_std + self.mean

        cache: dict[str, mx.array] = {}
        warm: dict[str, bool] = {}
        chunks: list[np.ndarray] = []

        for chunk_idx in range(int(x.shape[2])):
            z = x[:, :, chunk_idx : chunk_idx + 1, :, :]
            z = _causal_conv3d_ncthw(z, self.w["conv2.weight"], self.w.get("conv2.bias"))
            out = self._decode_chunk(z, first_chunk=(chunk_idx == 0), cache=cache, warm=warm)
            out = np.array(mx.clip(out.astype(mx.float32), -1.0, 1.0))
            finite_mask = np.isfinite(out)
            if not finite_mask.all():
                bad = int((~finite_mask).sum())
                out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
                if verbose:
                    print(
                        f"[wan22][vae] chunk {chunk_idx + 1}/{int(x.shape[2])} "
                        f"replaced non-finite values: {bad}",
                        flush=True,
                    )
            chunks.append(out)
            mx.clear_cache()

        return np.concatenate(chunks, axis=2)


def _build_sigma_schedule(num_steps: int, shift: float) -> list[float]:
    sigmas = np.linspace(1.0, 0.0, num_steps + 1, dtype=np.float32)[:-1]
    sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    out = sigmas.tolist()
    out.append(0.0)
    return [float(x) for x in out]


def _normalize_sampling_method(name: str) -> str:
    method = name.strip().lower()
    aliases = {
        "euler": "euler",
        "dpm++": "dpm++",
        "dpmpp": "dpm++",
        "dpm_plus_plus": "dpm++",
        "unipc": "unipc",
        "uni-pc": "unipc",
    }
    if method not in aliases:
        raise ValueError(f"Unsupported sampling method {name!r}. Expected one of: euler, dpm++, unipc.")
    return aliases[method]


def _flow_dpmpp_x0_pred(sample: mx.array, sigma: float, flow_pred: mx.array) -> mx.array:
    return sample - sigma * flow_pred


def _flow_dpmpp_first_order_update(sample: mx.array, model_output: mx.array, sigma_s: float, sigma_t: float) -> mx.array:
    if sigma_t == 0.0:
        return model_output
    alpha_t = 1.0 - sigma_t
    alpha_s = 1.0 - sigma_s
    with np.errstate(divide="ignore"):
        lambda_t = float(np.log(np.asarray(alpha_t, dtype=np.float64)) - np.log(np.asarray(sigma_t, dtype=np.float64)))
        lambda_s = float(np.log(np.asarray(alpha_s, dtype=np.float64)) - np.log(np.asarray(sigma_s, dtype=np.float64)))
    h = lambda_t - lambda_s
    return (sigma_t / sigma_s) * sample - (alpha_t * (math.exp(-h) - 1.0)) * model_output


def _flow_dpmpp_second_order_update(
    sample: mx.array,
    model_output_s0: mx.array,
    model_output_s1: mx.array,
    sigma_s1: float,
    sigma_s0: float,
    sigma_t: float,
) -> mx.array:
    alpha_t = 1.0 - sigma_t
    alpha_s0 = 1.0 - sigma_s0
    alpha_s1 = 1.0 - sigma_s1
    with np.errstate(divide="ignore"):
        lambda_t = float(np.log(np.asarray(alpha_t, dtype=np.float64)) - np.log(np.asarray(sigma_t, dtype=np.float64)))
        lambda_s0 = float(np.log(np.asarray(alpha_s0, dtype=np.float64)) - np.log(np.asarray(sigma_s0, dtype=np.float64)))
        lambda_s1 = float(np.log(np.asarray(alpha_s1, dtype=np.float64)) - np.log(np.asarray(sigma_s1, dtype=np.float64)))
    h = lambda_t - lambda_s0
    h0 = lambda_s0 - lambda_s1
    r0 = h0 / h
    d0 = model_output_s0
    d1 = (1.0 / r0) * (model_output_s0 - model_output_s1)
    coeff = alpha_t * (math.exp(-h) - 1.0)
    return (sigma_t / sigma_s0) * sample - coeff * d0 - 0.5 * coeff * d1


def _sigma_to_alpha_sigma_t(sigma: float) -> tuple[float, float]:
    return 1.0 - sigma, sigma


def _lambda_from_sigma(sigma: float) -> float:
    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma)
    with np.errstate(divide="ignore"):
        return float(np.log(np.asarray(alpha_t, dtype=np.float64)) - np.log(np.asarray(sigma_t, dtype=np.float64)))


def _flow_unipc_predict(
    sample: mx.array,
    model_outputs: list[mx.array],
    sigmas: list[float],
    step_idx: int,
    *,
    order: int,
) -> mx.array:
    sigma_t = sigmas[step_idx + 1]
    sigma_s0 = sigmas[step_idx]
    if sigma_t == 0.0:
        return model_outputs[-1]

    m0 = model_outputs[-1]
    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
    _, sigma_s0 = _sigma_to_alpha_sigma_t(sigma_s0)

    lambda_t = _lambda_from_sigma(sigmas[step_idx + 1])
    lambda_s0 = _lambda_from_sigma(sigmas[step_idx])
    h = lambda_t - lambda_s0
    hh = -h
    h_phi_1 = math.expm1(hh)
    b_h = math.expm1(hh)

    if order == 1 or len(model_outputs) < 2:
        return (sigma_t / sigma_s0) * sample - alpha_t * h_phi_1 * m0

    lambda_s1 = _lambda_from_sigma(sigmas[step_idx - 1])
    rk = (lambda_s1 - lambda_s0) / h
    d1 = (model_outputs[-2] - m0) / rk
    pred_res = 0.5 * d1
    x_t_base = (sigma_t / sigma_s0) * sample - alpha_t * h_phi_1 * m0
    return x_t_base - alpha_t * b_h * pred_res


def _flow_unipc_correct(
    this_model_output: mx.array,
    model_outputs: list[mx.array],
    sigmas: list[float],
    step_idx: int,
    *,
    last_sample: mx.array,
    this_sample: mx.array,
    order: int,
) -> mx.array:
    if order <= 0 or step_idx <= 0 or not model_outputs:
        return this_sample

    sigma_t = sigmas[step_idx]
    sigma_s0 = sigmas[step_idx - 1]
    m0 = model_outputs[-1]
    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
    _, sigma_s0 = _sigma_to_alpha_sigma_t(sigma_s0)
    lambda_t = _lambda_from_sigma(sigmas[step_idx])
    lambda_s0 = _lambda_from_sigma(sigmas[step_idx - 1])
    h = lambda_t - lambda_s0
    hh = -h
    h_phi_1 = math.expm1(hh)
    b_h = math.expm1(hh)

    x_t_base = (sigma_t / sigma_s0) * last_sample - alpha_t * h_phi_1 * m0
    d1_t = this_model_output - m0

    if order == 1 or len(model_outputs) < 2 or step_idx < 2:
        return x_t_base - alpha_t * b_h * (0.5 * d1_t)

    lambda_s1 = _lambda_from_sigma(sigmas[step_idx - 2])
    rk = (lambda_s1 - lambda_s0) / h
    d1 = (model_outputs[-2] - m0) / rk
    corr_res = 0.5 * d1
    return x_t_base - alpha_t * b_h * (corr_res + 0.5 * d1_t)


def _parse_compare_samplers(value: str | None, default_method: str) -> list[str]:
    if value is None or not value.strip():
        return [default_method]
    raw = value.strip().lower()
    if raw == "all":
        parts = ["euler", "dpm++", "unipc"]
    else:
        parts = [part.strip() for part in value.split(",") if part.strip()]
    methods: list[str] = []
    seen: set[str] = set()
    for part in parts:
        method = _normalize_sampling_method(part)
        if method not in seen:
            seen.add(method)
            methods.append(method)
    if not methods:
        return [default_method]
    return methods


def _sample_latents(
    method: str,
    *,
    seed: int,
    latent_shape: tuple[int, ...],
    sigmas: list[float],
    transformer: "WanTransformer",
    context: mx.array,
    context_null: mx.array,
    cfg_scale: float,
    verbose: bool,
) -> mx.array:
    latents = _sample_noise(seed, latent_shape, mx.float32)
    dpmpp_history: list[mx.array] = []
    unipc_history: list[mx.array] = []
    unipc_last_sample: mx.array | None = None
    unipc_last_order = 1

    _print_verbose(verbose, f"[wan22][{method}] begin sampling")

    for step_idx, sigma in enumerate(sigmas[:-1]):
        sigma_next = sigmas[step_idx + 1]
        step_t0 = time.perf_counter()
        pred_cond = transformer.forward(latents, sigma, context)
        pred_uncond = transformer.forward(latents, sigma, context_null)
        pred = pred_uncond.astype(mx.float32) + cfg_scale * (
            pred_cond.astype(mx.float32) - pred_uncond.astype(mx.float32)
        )

        if method == "euler":
            latents = latents + (sigma_next - sigma) * pred
        elif method == "dpm++":
            x0_pred = _flow_dpmpp_x0_pred(latents, sigma, pred)
            dpmpp_history.append(x0_pred)
            if len(dpmpp_history) > 2:
                del dpmpp_history[0]
            use_first_order = step_idx == 0 or step_idx == len(sigmas) - 2
            if use_first_order or len(dpmpp_history) < 2:
                latents = _flow_dpmpp_first_order_update(latents, dpmpp_history[-1], sigma, sigma_next)
            else:
                latents = _flow_dpmpp_second_order_update(
                    latents,
                    dpmpp_history[-1],
                    dpmpp_history[-2],
                    sigmas[step_idx - 1],
                    sigma,
                    sigma_next,
                )
        elif method == "unipc":
            x0_pred = _flow_dpmpp_x0_pred(latents, sigma, pred)
            if step_idx > 0 and unipc_last_sample is not None:
                latents = _flow_unipc_correct(
                    x0_pred,
                    unipc_history,
                    sigmas,
                    step_idx,
                    last_sample=unipc_last_sample,
                    this_sample=latents,
                    order=unipc_last_order,
                )
            unipc_history.append(x0_pred)
            if len(unipc_history) > 2:
                del unipc_history[0]
            remaining = len(sigmas) - 1 - step_idx
            this_order = min(2, remaining, step_idx + 1)
            unipc_last_sample = latents
            latents = _flow_unipc_predict(latents, unipc_history, sigmas, step_idx, order=this_order)
            unipc_last_order = this_order
        else:
            raise ValueError(f"Unsupported sampler {method!r}.")

        mx.eval(latents)
        _print_array_stats(f"{method}.pred.step_{step_idx + 1:02d}", pred, enabled=verbose)
        _print_array_stats(f"{method}.latents.step_{step_idx + 1:02d}", latents, enabled=verbose)
        step_t1 = time.perf_counter()
        _print_verbose(
            verbose,
            f"[wan22][{method}] step {step_idx + 1:02d}/{len(sigmas) - 1} sigma={sigma:.4f}->{sigma_next:.4f} "
            f"{step_t1 - step_t0:.2f}s",
        )

    _print_array_stats(f"{method}.latents.after_sampling", latents, enabled=verbose)
    return latents


def _check_quantized_backend() -> tuple[bool, str | None]:
    try:
        w = mx.arange(32 * 32, dtype=mx.float16).reshape((32, 32))
        qw, qs, qb = mx.quantize(w, group_size=32, bits=4)
        x = mx.ones((1, 32), dtype=mx.float16)
        y = mx.quantized_matmul(
            x,
            qw,
            qs,
            qb,
            transpose=True,
            group_size=32,
            bits=4,
            mode="affine",
        )
        mx.eval(y)
        return True, None
    except Exception as exc:
        return False, repr(exc)


def _sample_noise(seed: int, shape: tuple[int, ...], dtype: mx.Dtype) -> mx.array:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(shape, dtype=np.float32)
    return mx.array(noise).astype(dtype)


def _video_to_uint8_frames(video: mx.array | np.ndarray) -> list[np.ndarray]:
    if isinstance(video, np.ndarray):
        arr = np.clip((video.astype(np.float32) + 1.0) * 127.5, 0.0, 255.0)
        frames = np.transpose(arr, (1, 2, 3, 0))
    else:
        video = mx.clip((video.astype(mx.float32) + 1.0) * 127.5, 0.0, 255.0)
        frames = np.array(video.transpose((1, 2, 3, 0)))
    if not np.isfinite(frames).all():
        frames = np.nan_to_num(frames, nan=0.0, posinf=255.0, neginf=0.0)
    return [frame.astype(np.uint8) for frame in frames]


def _make_compare_frame(frames: list[np.ndarray], labels: list[str]) -> np.ndarray:
    if not frames:
        raise ValueError("No frames to compare.")
    width = sum(frame.shape[1] for frame in frames)
    height = max(frame.shape[0] for frame in frames)
    banner_h = 40
    canvas = Image.new("RGB", (width, height + banner_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for frame, label in zip(frames, labels):
        image = Image.fromarray(frame, mode="RGB")
        canvas.paste(image, (x, banner_h))
        draw.rectangle((x, 0, x + frame.shape[1], banner_h), fill=(30, 30, 30))
        text_box = draw.textbbox((0, 0), label)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        draw.text(
            (x + (frame.shape[1] - text_w) / 2, (banner_h - text_h) / 2 - 1),
            label,
            fill=(235, 235, 235),
        )
        x += frame.shape[1]
    return np.array(canvas)


def _make_comparison_frames(frames_by_method: dict[str, list[np.ndarray]], methods: list[str]) -> list[np.ndarray]:
    lengths = {len(frames_by_method[method]) for method in methods}
    if len(lengths) != 1:
        raise ValueError(f"Sampler outputs have mismatched frame counts: {sorted(lengths)}")
    num_frames = lengths.pop()
    return [
        _make_compare_frame([frames_by_method[method][idx] for method in methods], methods)
        for idx in range(num_frames)
    ]


def _print_verbose(enabled: bool, text: str) -> None:
    if enabled:
        print(text, flush=True)


def _print_array_stats(name: str, x: mx.array | np.ndarray, *, enabled: bool) -> None:
    if not enabled:
        return
    arr = np.array(x, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        print(f"[wan22] {name}: all values are non-finite", flush=True)
        return
    valid = arr[finite]
    print(
        f"[wan22] {name}: shape={arr.shape} finite={int(finite.sum())}/{arr.size} "
        f"min={valid.min():.6f} max={valid.max():.6f} mean={valid.mean():.6f}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wan2.2 TI2V GGUF inference with local MLX.")
    parser.add_argument("-M", "--mode", default="vid_gen")
    parser.add_argument("--diffusion-model", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--t5xxl", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("-p", "--prompt", required=True)
    parser.add_argument("-n", "--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("-W", "--width", type=int, default=480)
    parser.add_argument("-H", "--height", type=int, default=832)
    parser.add_argument("--video-frames", type=int, default=33)
    parser.add_argument("--cfg-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--sampling-method", default="euler")
    parser.add_argument("--compare-samplers", default=None)
    parser.add_argument("-s", "--steps", "--sampling-steps", dest="sampling_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--vae-device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--vae-dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--output", default="output/wan22.mp4")
    parser.add_argument("--offload-to-cpu", action="store_true")
    parser.add_argument("--diffusion-fa", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sampling_method = _normalize_sampling_method(args.sampling_method)
    sampling_methods = _parse_compare_samplers(args.compare_samplers, sampling_method)
    if args.mode != "vid_gen":
        raise ValueError(f"Only `vid_gen` mode is supported, got {args.mode!r}.")
    if args.width % 32 != 0 or args.height % 32 != 0:
        raise ValueError("Width and height must both be divisible by 32.")
    if args.video_frames < 1 or (args.video_frames - 1) % 4 != 0:
        raise ValueError("--video-frames must be of the form 4n+1.")
    if args.sampling_steps <= 0:
        raise ValueError("--sampling-steps must be > 0.")

    if args.offload_to_cpu:
        mx.set_default_device(mx.cpu)

    quant_backend_ok, quant_backend_error = _check_quantized_backend()
    if not quant_backend_ok:
        raise RuntimeError(
            "This MLX environment cannot execute GGUF quantized ops on the current backend. "
            f"default_device={mx.default_device()} error={quant_backend_error}"
        )

    seed = args.seed if args.seed >= 0 else random.randint(0, 2**31 - 1)
    _print_verbose(args.verbose, f"[wan22] seed={seed}")

    t0 = time.perf_counter()
    wan_weights, wan_meta = _load_gguf(args.diffusion_model)
    t1 = time.perf_counter()
    _print_verbose(args.verbose, f"[wan22] loaded diffusion model in {t1 - t0:.2f}s")

    t5_weights, t5_meta = _load_gguf(args.t5xxl)
    t2 = time.perf_counter()
    _print_verbose(args.verbose, f"[wan22] loaded T5 encoder in {t2 - t1:.2f}s")

    tokenizer = GGUFSentencePieceTokenizer.from_gguf(args.t5xxl if args.tokenizer is None else args.tokenizer)
    text_encoder = UMT5Encoder(t5_weights, t5_meta)
    transformer = WanTransformer(wan_weights, wan_meta)
    t3 = time.perf_counter()

    prompt_ids = tokenizer.encode(args.prompt, max_length=text_encoder.text_len)
    negative_ids = tokenizer.encode(args.negative_prompt, max_length=text_encoder.text_len)
    _print_verbose(
        args.verbose,
        f"[wan22] prompt_tokens={len(prompt_ids)} negative_tokens={len(negative_ids)}",
    )

    context = text_encoder.encode(prompt_ids)
    context_null = text_encoder.encode(negative_ids)
    mx.eval(context, context_null)
    t4 = time.perf_counter()
    _print_verbose(args.verbose, f"[wan22] encoded text in {t4 - t3:.2f}s")

    latent_t = (args.video_frames - 1) // 4 + 1
    latent_h = args.height // 16
    latent_w = args.width // 16
    latent_shape = (1, 48, latent_t, latent_h, latent_w)
    sigmas = _build_sigma_schedule(args.sampling_steps, args.flow_shift)
    _print_verbose(
        args.verbose,
        f"[wan22] latent_shape={latent_shape} steps={args.sampling_steps} shift={args.flow_shift} "
        f"samplers={','.join(sampling_methods)} vae_device={args.vae_device}",
    )
    if len(sampling_methods) > 1:
        _print_verbose(args.verbose, f"[wan22] compare_samplers={','.join(sampling_methods)}")

    sampled_latents: dict[str, mx.array] = {}
    for method in sampling_methods:
        t_sample0 = time.perf_counter()
        sampled_latents[method] = _sample_latents(
            method,
            seed=seed,
            latent_shape=latent_shape,
            sigmas=sigmas,
            transformer=transformer,
            context=context,
            context_null=context_null,
            cfg_scale=args.cfg_scale,
            verbose=args.verbose,
        )
        t_sample1 = time.perf_counter()
        _print_verbose(args.verbose, f"[wan22][{method}] sampled latents in {t_sample1 - t_sample0:.2f}s")

    del context, context_null, text_encoder, transformer, wan_weights, wan_meta, t5_weights, t5_meta
    mx.clear_cache()

    t_vae0 = time.perf_counter()
    vae = _build_vae_decoder(args.vae, args.vae_device, args.vae_dtype)
    t_vae1 = time.perf_counter()
    _print_verbose(
        args.verbose,
        f"[wan22] loaded VAE on {args.vae_device} dtype={args.vae_dtype} in {t_vae1 - t_vae0:.2f}s",
    )

    frames_by_method: dict[str, list[np.ndarray]] = {}
    for method in sampling_methods:
        latents = sampled_latents[method]
        if args.vae_device == "cpu":
            latents = _copy_array_to_device(latents, "cpu")
            mx.clear_cache()
        t_decode0 = time.perf_counter()
        video = vae.decode(latents, verbose=args.verbose)[0]
        t_decode1 = time.perf_counter()
        _print_verbose(args.verbose, f"[wan22][{method}] decoded video in {t_decode1 - t_decode0:.2f}s")
        frames_by_method[method] = _video_to_uint8_frames(video)
        del video
        mx.clear_cache()

    if len(sampling_methods) > 1:
        frames = _make_comparison_frames(frames_by_method, sampling_methods)
    else:
        frames = frames_by_method[sampling_methods[0]]

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(output_path), frames, fps=args.fps)
    t7 = time.perf_counter()

    if len(sampling_methods) > 1:
        print(
            f"saved {output_path} frames={len(frames)} size={frames[0].shape[1]}x{frames[0].shape[0]} "
            f"seed={seed} samplers={','.join(sampling_methods)}"
        )
    else:
        print(f"saved {output_path} frames={len(frames)} size={args.width}x{args.height} seed={seed}")
    _print_verbose(args.verbose, f"[wan22] total_time={t7 - t0:.2f}s")


if __name__ == "__main__":
    main()
