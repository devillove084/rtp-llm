"""Transport-independent RTP hidden-state keys, JSON v1 markers and FP8 layout.

The key and metadata APIs require only the Python standard library. Tensor
payloads are not part of the JSON document; shapes and dtypes describe the
actual stored tensors, including any quantization performed by the producer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


MAX_CAPTURE_METADATA_BYTES = 1 << 16
_MAX_REQUEST_ID = (1 << 63) - 1
_NAMESPACE = r"[A-Za-z0-9]+"
_KEY = re.compile(rf"rtp_{_NAMESPACE}_(0|[1-9][0-9]{{0,18}})")
_MARKER_FIELDS = {"capture_layer_ids", "tensor_shapes", "tensor_dtypes", "version"}
_BF16_LAYOUT = {"input_ids": "int64", "last_hidden_states": "bfloat16", "hidden_states": "bfloat16"}


def make_store_key(namespace: str, request_id: int) -> str:
    """Return the canonical capture key for an ASCII namespace and nonnegative int64 ID."""
    if not isinstance(request_id, int) or isinstance(request_id, bool):
        raise TypeError("request_id must be an integer")
    if not 0 <= request_id <= _MAX_REQUEST_ID:
        raise ValueError("request_id out of int64 capture range")
    if not isinstance(namespace, str) or re.fullmatch(_NAMESPACE, namespace) is None:
        raise ValueError("namespace must contain one or more ASCII letters or digits")
    return f"rtp_{namespace}_{request_id}"


def is_store_key(key: object) -> bool:
    match = _KEY.fullmatch(key) if isinstance(key, str) else None
    return match is not None and int(match[1]) <= _MAX_REQUEST_ID


def validate_capture_metadata(metadata: Any) -> dict[str, Any]:
    """Validate a decoded marker, preserving its tensor and auxiliary-layer ordering."""
    if not isinstance(metadata, dict) or metadata.keys() != _MARKER_FIELDS:
        raise ValueError("unexpected JSON v1 marker fields")
    version = metadata["version"]
    if type(version) is not int or version != 1:
        raise ValueError("expected integer capture version 1")

    layers = metadata["capture_layer_ids"]
    if not isinstance(layers, list) or not layers or not all(type(i) is int and i >= 0 for i in layers):
        raise ValueError("expected ordered nonnegative integer layer IDs")
    if len(set(layers)) != len(layers):
        raise ValueError("duplicate capture layer ID")

    shapes = metadata["tensor_shapes"]
    types = metadata["tensor_dtypes"]
    if not isinstance(shapes, dict) or not isinstance(types, dict):
        raise ValueError("tensor shapes and dtypes must be objects")
    layout = _BF16_LAYOUT.copy()
    if "target" in shapes:
        layout["target"] = "bfloat16"
    if "hidden_states_scale" in shapes:
        layout.update(hidden_states="float8_e4m3fn", hidden_states_scale="float32")
    if shapes.keys() != layout.keys() or types != layout:
        raise ValueError("tensor names or dtypes do not match a supported capture layout")

    for field, dims in shapes.items():
        if (not isinstance(dims, list) or len(dims) != (1 if field == "input_ids" else 2)
                or not all(type(n) is int and n > 0 for n in dims)):
            raise ValueError(f"invalid tensor dimensions for {field}")
    tokens, width = shapes["last_hidden_states"]
    if any(dims[0] != tokens for dims in shapes.values()):
        raise ValueError("inconsistent capture token counts")
    if shapes["hidden_states"][1] != width * len(layers):
        raise ValueError("auxiliary width does not cover the declared layers")
    if "hidden_states_scale" in shapes and shapes["hidden_states_scale"][1] != len(layers):
        raise ValueError("expected one scale per token and auxiliary layer")
    return metadata


def _check_size(payload: bytes) -> None:
    if len(payload) > MAX_CAPTURE_METADATA_BYTES:
        raise ValueError("capture marker exceeds the 64 KiB JSON v1 limit")


def encode_capture_metadata(capture_metadata: Mapping, actual_metadata: Mapping) -> bytes:
    """Encode ordered layer IDs and a post-conversion shapes/dtypes receipt."""
    if not isinstance(capture_metadata, Mapping) or capture_metadata.keys() != {"capture_layer_ids"}:
        raise ValueError("expected capture_layer_ids as the sole capture parameter")
    document = dict(version=1, capture_layer_ids=capture_metadata["capture_layer_ids"],
                    tensor_shapes={k: list(v) for k, v in actual_metadata["shapes"].items()},
                    tensor_dtypes={k: str(v).removeprefix("torch.") for k, v in actual_metadata["dtypes"].items()})
    encoded = json.dumps(validate_capture_metadata(document), separators=(",", ":"), allow_nan=False).encode("utf-8")
    _check_size(encoded)
    return encoded


def _json_object(items):
    obj = dict(items)
    if len(obj) != len(items):
        raise ValueError("duplicate JSON member in capture marker")
    return obj


def decode_capture_metadata(payload: bytes) -> dict[str, Any]:
    """Decode a bounded UTF-8 marker; reject duplicate members at any depth."""
    _check_size(payload)
    return validate_capture_metadata(json.loads(payload.decode("utf-8"), object_pairs_hook=_json_object))


def quantize_aux_hidden_states(hidden_states: torch.Tensor, num_layers: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode [..., L*H] as E4M3FN plus FP32 scales [..., L], on the input device."""
    import torch

    if num_layers <= 0 or hidden_states.size(-1) % num_layers:
        raise ValueError("expected a positive layer count dividing the auxiliary width")
    if not hidden_states.is_floating_point():
        raise TypeError("auxiliary hidden states must be floating point")

    codes = torch.empty_like(hidden_states, dtype=torch.float8_e4m3fn)
    bound = torch.finfo(codes.dtype).max
    width = hidden_states.size(-1) // num_layers
    vectors = hidden_states.unflatten(-1, (num_layers, width))
    factors = hidden_states.new_empty(vectors.shape[:-1], dtype=torch.float32)
    # Converting the entire concatenation to FP32 would multiply capture's
    # working set. The final dimension of each view is exactly one layer.
    for index in range(num_layers):
        values = vectors.select(-2, index).float()
        magnitude = values.abs().amax(dim=-1)
        factor = magnitude.clamp_min_(1e-12).div_(bound)
        normalized = values / factor.unsqueeze(-1)
        codes.narrow(-1, index * width, width).copy_(normalized.clamp_(-bound, bound))
        factors.select(-1, index).copy_(factor)
    return codes, factors
