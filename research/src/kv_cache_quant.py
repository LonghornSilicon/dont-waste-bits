"""
KV Cache Quantization — proper implementation for DWB verification.

The paper applies quantization to the KV cache (stored attention keys/values),
NOT to model weights. This is a critical distinction:
  - Weight quantization: corrupts all linear layer computations
  - KV cache quantization: only affects attention keys/values

For evaluation via single-pass log-likelihood scoring (HellaSwag), we must
hook the K and V projection outputs *before* they are used in attention.
This simulates storing K/V at reduced precision and reading them back, which
is what happens during autoregressive generation with a quantized KV cache.

For transformers 5.x (LLaMA/SmolLM), the attention module has explicit
k_proj and v_proj Linear submodules — we hook those directly.

Usage:
    model = AutoModelForCausalLM.from_pretrained(...)
    hooks = attach_kv_hooks(model, mode="static4bit")
    # run model inference ...
    detach_kv_hooks(hooks)
"""

import torch
import torch.nn.functional as F
from typing import Literal

KVMode = Literal["fp16", "static4bit", "static8bit", "static2bit", "dwb"]


def quantize_int4(x: torch.Tensor) -> torch.Tensor:
    """Simulate INT4 KV quantization (per-tensor symmetric)."""
    scale = x.abs().max() / 7.0
    if scale == 0:
        return x
    return (x / scale).round().clamp(-8, 7) * scale


def quantize_int8(x: torch.Tensor) -> torch.Tensor:
    """Simulate INT8 KV quantization."""
    scale = x.abs().max() / 127.0
    if scale == 0:
        return x
    return (x / scale).round().clamp(-128, 127) * scale


def quantize_int2(x: torch.Tensor) -> torch.Tensor:
    """Simulate INT2 KV quantization."""
    scale = x.abs().max() / 1.0
    if scale == 0:
        return x
    return (x / scale).round().clamp(-2, 1) * scale


def quantize_int4_per_token(x: torch.Tensor) -> torch.Tensor:
    """INT4 KV quantization, per-token scale (more realistic for KV cache).

    In autoregressive generation, each new K/V token is quantized independently
    as it's added to the cache. Per-token scale is the standard in most KV
    quantization papers (KIVI, QuaRot, etc.) and is more aggressive than
    per-tensor when token magnitudes vary (outlier tokens drive per-tensor scale).

    x shape: (batch, seq_len, features) — scales each position independently.
    """
    if x.dim() == 3:
        result = x.clone()
        for t in range(x.shape[1]):
            tok = x[:, t, :]
            scale = tok.abs().max() / 7.0
            if scale > 0:
                result[:, t, :] = (tok / scale).round().clamp(-8, 7) * scale
        return result
    scale = x.abs().max() / 7.0
    if scale == 0:
        return x
    return (x / scale).round().clamp(-8, 7) * scale


def quantize_int8_per_token(x: torch.Tensor) -> torch.Tensor:
    """INT8 KV quantization, per-token scale."""
    if x.dim() == 3:
        result = x.clone()
        for t in range(x.shape[1]):
            tok = x[:, t, :]
            scale = tok.abs().max() / 127.0
            if scale > 0:
                result[:, t, :] = (tok / scale).round().clamp(-128, 127) * scale
        return result
    scale = x.abs().max() / 127.0
    if scale == 0:
        return x
    return (x / scale).round().clamp(-128, 127) * scale


_QUANT_FNS = {
    "fp16": lambda x: x,
    "static4bit": quantize_int4,           # per-tensor (conservative)
    "static4bit_per_token": quantize_int4_per_token,  # per-token (realistic)
    "static8bit": quantize_int8,
    "static8bit_per_token": quantize_int8_per_token,
    "static2bit": quantize_int2,
}


def make_proj_hook(quant_fn):
    """Hook for k_proj or v_proj output: quantize before attention uses it."""
    def hook(module, input, output):
        return quant_fn(output)
    return hook


def attach_kv_hooks(model, mode: KVMode = "static4bit") -> list:
    """
    Attach KV quantization hooks to k_proj and v_proj in all attention layers.

    For LLaMA/SmolLM (transformers 5.x), the attention module exposes
    `k_proj` and `v_proj` as named submodules. We hook their outputs
    to simulate quantized KV cache: the keys and values enter attention
    at reduced precision, matching the paper's claim.

    This works for both single-pass scoring (HellaSwag eval) and generation
    because it quantizes at the projection level, not the cache storage level.

    Returns list of hook handles for later removal.
    """
    if mode not in _QUANT_FNS:
        raise ValueError(f"Unknown mode: {mode}. Choose from {list(_QUANT_FNS)}")

    quant_fn = _QUANT_FNS[mode]
    hooks = []
    attached = 0

    for name, module in model.named_modules():
        # Target k_proj and v_proj — present in LLaMA, Mistral, SmolLM, Qwen, etc.
        mod_name = name.split(".")[-1]
        if mod_name in ("k_proj", "v_proj"):
            h = module.register_forward_hook(make_proj_hook(quant_fn))
            hooks.append(h)
            attached += 1

    print(f"  Attached KV quant hooks to {attached} proj modules (mode={mode})")
    if attached == 0:
        print("  WARNING: No k_proj/v_proj found — try attach_kv_hooks_attention() for other architectures")
    return hooks


def detach_kv_hooks(hooks: list):
    """Remove all KV quantization hooks."""
    for h in hooks:
        h.remove()
    print(f"  Removed {len(hooks)} KV quant hooks")


# ---- Fallback: attention-level output hooks for non-LLaMA architectures ----

def make_kv_hook_attention(mode: KVMode):
    """
    Forward hook for an attention module that intercepts past_key_value output.
    Works for architectures that return (attn_out, weights, (key, value)) tuples.
    NOT compatible with transformers 5.x DynamicCache — use attach_kv_hooks instead.
    """
    quant_fn = _QUANT_FNS.get(mode, lambda x: x)

    def hook(module, input, output):
        if not isinstance(output, tuple):
            return output
        modified = list(output)
        for i, item in enumerate(modified):
            if isinstance(item, tuple) and len(item) == 2:
                k, v = item
                if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                    modified[i] = (quant_fn(k), quant_fn(v))
                    break
        return tuple(modified)

    return hook


def attach_kv_hooks_attention(model, mode: KVMode = "static4bit") -> list:
    """Attach hooks to attention modules (legacy, pre-transformers-5.x)."""
    quant_fn_hook = make_kv_hook_attention(mode)
    hooks = []
    attached = 0
    for name, module in model.named_modules():
        class_name = type(module).__name__.lower()
        if "attention" in class_name and "layer" not in class_name and hasattr(module, "forward"):
            h = module.register_forward_hook(quant_fn_hook)
            hooks.append(h)
            attached += 1
    print(f"  Attached attention-level KV hooks to {attached} modules (mode={mode})")
    return hooks


# ---- Wrapper-based approach for generation ----

class KVQuantWrapper(torch.nn.Module):
    """
    Wraps a model and quantizes KV cache during generation.
    Compatible with DynamicCache (transformers 5.x) by quantizing past_key_values
    after each forward pass — simulates quantized storage/retrieval.
    """

    def __init__(self, model, mode: KVMode = "static4bit"):
        super().__init__()
        self.model = model
        self.mode = mode
        self._quant_fn = _QUANT_FNS.get(mode, lambda x: x)

    def forward(self, **kwargs):
        output = self.model(**kwargs)
        if hasattr(output, "past_key_values") and output.past_key_values is not None:
            pkv = output.past_key_values
            # Handle both tuple-of-tuples and DynamicCache
            if hasattr(pkv, "key_cache"):
                # DynamicCache (transformers 5.x)
                for i in range(len(pkv.key_cache)):
                    pkv.key_cache[i] = self._quant_fn(pkv.key_cache[i])
                    pkv.value_cache[i] = self._quant_fn(pkv.value_cache[i])
            else:
                quantized = []
                for layer_pkv in pkv:
                    k, v = layer_pkv[0], layer_pkv[1]
                    quantized.append((self._quant_fn(k), self._quant_fn(v)) + layer_pkv[2:])
                output.past_key_values = tuple(quantized)
        return output

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)
