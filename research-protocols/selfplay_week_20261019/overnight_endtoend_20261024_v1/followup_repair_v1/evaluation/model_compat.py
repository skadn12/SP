"""Checked, in-memory config bridge for this Qwen3 default-RoPE replay.

Importing this module imports only the standard library. No model weights are
read, no CUDA API is called, and the checkpoint config is never rewritten.
This intentionally rejects scaling, sliding attention, and ambiguous schemas.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path


class ConfigCompatibilityError(ValueError):
    """The checkpoint falls outside this explicitly checked bridge."""


_INTEGER_FIELDS = (
    "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim",
    "max_position_embeddings", "max_window_layers",
)
_SEMANTIC_FIELDS = _INTEGER_FIELDS + (
    "architectures", "model_type", "hidden_act", "attention_bias",
    "attention_dropout", "rms_norm_eps", "initializer_range",
    "tie_word_embeddings", "bos_token_id", "eos_token_id",
)


def _require(condition, message):
    if not condition:
        raise ConfigCompatibilityError(message)


def _number(value, field):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{field} must be a number")
    _require(math.isfinite(value) and value > 0, f"{field} must be finite and positive")
    return value


def _dtype(value, field):
    _require(isinstance(value, str), f"{field} must be an explicit dtype string")
    value = value.removeprefix("torch.")
    _require(value == "bfloat16", f"{field}: this replay supports only bfloat16")
    return value


def _read_and_validate(directory):
    directory = Path(directory).resolve(strict=True)
    path = directory / "config.json"
    payload = path.read_bytes()
    raw = json.loads(payload)
    _require(isinstance(raw, dict), "config.json must contain an object")
    _require(raw.get("model_type") == "qwen3", "Only Qwen3 is supported")
    _require(raw.get("architectures") == ["Qwen3ForCausalLM"],
             "Expected the Qwen3ForCausalLM architecture")
    for field in _INTEGER_FIELDS:
        value = raw.get(field)
        _require(type(value) is int and value > 0, f"{field} must be an explicit positive integer")
    _require(raw["head_dim"] % 2 == 0, "head_dim must be even for default RoPE")
    _require(raw["num_attention_heads"] % raw["num_key_value_heads"] == 0,
             "Attention heads must divide evenly into key/value head groups")
    _require(raw["max_window_layers"] <= raw["num_hidden_layers"],
             "max_window_layers exceeds num_hidden_layers")
    for field in ("attention_bias", "tie_word_embeddings", "use_cache"):
        _require(type(raw.get(field)) is bool, f"{field} must be an explicit boolean")
    _require(raw.get("hidden_act") == "silu", "Only the present silu Qwen3 config is supported")
    _require(raw.get("attention_dropout") == 0.0 and not isinstance(raw.get("attention_dropout"), bool),
             "This replay requires zero attention dropout")
    for field in ("rms_norm_eps", "initializer_range"):
        _number(raw.get(field), field)
    for field in ("bos_token_id", "eos_token_id"):
        value = raw.get(field)
        _require(type(value) is int and 0 <= value < raw["vocab_size"],
                 f"{field} must be an explicit token id inside the vocabulary")
    pad = raw.get("pad_token_id")
    _require(pad is None or (type(pad) is int and 0 <= pad < raw["vocab_size"]),
             "pad_token_id must be null or a valid token id")

    _require(raw.get("use_sliding_window") is False,
             "Only explicitly disabled sliding attention is supported")
    _require(raw.get("sliding_window") is None, "A non-null sliding_window is unsupported")
    layers = raw.get("layer_types")
    _require(layers is None or layers == ["full_attention"] * raw["num_hidden_layers"],
             "Only full_attention layers are supported")
    _require(raw.get("rope_scaling") is None, "Non-null rope_scaling is unsupported")

    top_present = "rope_theta" in raw
    nested = raw.get("rope_parameters")
    _require(nested is None or isinstance(nested, dict), "rope_parameters must be an object or null")
    if nested is not None:
        _require(set(nested) == {"rope_type", "rope_theta"},
                 "Only explicit default rope_type and rope_theta may occur in rope_parameters")
        _require(nested["rope_type"] == "default", "Only default RoPE is supported")
        _number(nested["rope_theta"], "rope_parameters.rope_theta")
    _require(top_present or nested is not None, "An explicit RoPE theta is required; defaults are forbidden")
    if top_present:
        _number(raw["rope_theta"], "rope_theta")
    if top_present and nested is not None:
        _require(raw["rope_theta"] == nested["rope_theta"],
                 "Top-level and nested rope_theta conflict")
    theta = raw["rope_theta"] if top_present else nested["rope_theta"]

    dtype_fields = {field: _dtype(raw[field], field)
                    for field in ("dtype", "torch_dtype") if field in raw}
    _require(bool(dtype_fields), "An explicit dtype or torch_dtype is required")
    _require(len(set(dtype_fields.values())) == 1, "dtype and torch_dtype conflict")
    dtype = next(iter(dtype_fields.values()))
    source = {
        "directory": str(directory), "config_path": str(path),
        "config_sha256": hashlib.sha256(payload).hexdigest(), "config_bytes": len(payload),
        "saved_transformers_version": raw.get("transformers_version"),
        "theta_fields": {key: raw[key] for key in ("rope_theta", "rope_parameters") if key in raw},
        "dtype_fields": {key: raw[key] for key in dtype_fields},
    }
    return raw, theta, dtype, source


def load_compatible_config(model_dir, reference_dir=None):
    """Return ``(config, receipt)`` after explicit, fail-closed validation.

    ``reference_dir`` optionally binds the source architecture, dtype and RoPE
    semantics to a second checkpoint config, such as the original Qwen3-4B base.
    This function loads config only. Passing its result as ``config=`` to the
    model loader is required; ordinary reloading of the source path loses it.
    """
    raw, theta, dtype, source = _read_and_validate(model_dir)
    reference = None
    if reference_dir is not None:
        reference_raw, reference_theta, reference_dtype, reference = _read_and_validate(reference_dir)
        for field in _SEMANTIC_FIELDS:
            _require(raw.get(field) == reference_raw.get(field), f"Reference mismatch: {field}")
        _require(theta == reference_theta, "Reference mismatch: rope_theta")
        _require(dtype == reference_dtype, "Reference mismatch: dtype")
        _require(raw.get("pad_token_id") == reference_raw.get("pad_token_id"),
                 "Reference mismatch: pad_token_id")

    normalized = copy.deepcopy(raw)
    normalized.pop("model_type")
    normalized["rope_theta"] = theta
    normalized["rope_scaling"] = None
    normalized["rope_parameters"] = {"rope_type": "default", "rope_theta": theta}
    normalized["dtype"] = dtype
    normalized["torch_dtype"] = dtype
    normalized["layer_types"] = ["full_attention"] * raw["num_hidden_layers"]
    # Lazy import: importing model_compat itself cannot initialize Torch/CUDA.
    from transformers import AutoConfig
    config = AutoConfig.for_model("qwen3", **normalized)
    # Transformers 4 consumes the top-level representation; Transformers 5 uses
    # rope_parameters. Keep both explicit in memory after construction.
    config.rope_theta = theta
    config.rope_scaling = None
    config.rope_parameters = {"rope_type": "default", "rope_theta": theta}
    _require(config.rope_theta == theta and config.rope_parameters["rope_theta"] == theta,
             "Runtime config did not retain the checked theta")
    _require(config.use_sliding_window is False and config.sliding_window is None,
             "Runtime attention config drift")
    _require(config.layer_types == normalized["layer_types"], "Runtime layer_types drift")
    runtime_dtype = getattr(config, "torch_dtype", None)
    _require(str(runtime_dtype).removeprefix("torch.") == dtype, "Runtime dtype drift")
    for field in _SEMANTIC_FIELDS:
        _require(getattr(config, field, None) == raw.get(field), f"Runtime field drift: {field}")

    receipt = {
        "schema_version": "qwen3_default_rope_config_bridge_v1", "status": "PASS",
        "source": source, "reference": reference,
        "runtime_transformers_version": importlib.metadata.version("transformers"),
        "effective_rope_theta": theta, "effective_rope_type": "default",
        "effective_dtype": dtype, "use_sliding_window": False,
        "architecture_fields_checked": list(_SEMANTIC_FIELDS),
        "normalization": {
            "top_level_theta_added": "rope_theta" not in raw,
            "nested_default_rope_added": raw.get("rope_parameters") is None,
            "dtype_aliases_normalized": True, "all_full_attention_explicit": True,
        },
        "checkpoint_files_modified": False, "weights_loaded": False,
        "cuda_called": False,
    }
    return config, receipt
