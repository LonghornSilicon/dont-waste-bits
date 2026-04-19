"""
Self-contained PolarQuant implementation (TurboQuant key component).

TurboQuant (ICLR 2026, Google): vector quantization for KV cache via random rotation.
Reference: "TurboQuant: Redefining AI Efficiency with Extreme Compression"

PolarQuant algorithm:
  1. Apply random Hadamard-like rotation: x' = R @ x, where R is an orthogonal matrix
     (random sign flip + Walsh-Hadamard transform, or random Gaussian rotation)
  2. Scalar quantize x' to INT-b (b=3 for keys, b=2 for values)
  3. Dequantize: x_hat = R^T @ dequant(x')

Key property: the rotation decorrelates the input dimensions and makes the distribution
more uniform (closer to isotropic Gaussian), which improves per-channel scalar quantization.

This is a CPU-friendly approximation of TurboQuant without external dependencies.
"""

import torch
import math


def _hadamard_transform(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform (WHT) on last dimension.
    Requires x.shape[-1] to be a power of 2 — pad if needed.
    """
    d = x.shape[-1]
    assert d & (d - 1) == 0, f"Hadamard requires power-of-2 dim, got {d}"
    h = x.clone()
    step = 1
    while step < d:
        for i in range(0, d, step * 2):
            a = h[..., i:i + step].clone()
            b = h[..., i + step:i + step * 2].clone()
            h[..., i:i + step] = a + b
            h[..., i + step:i + step * 2] = a - b
        step *= 2
    if normalize:
        h = h / math.sqrt(d)
    return h


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


class PolarQuant:
    """
    PolarQuant: random rotation + scalar quantization for KV vectors.

    Args:
        bits: quantization bit-width (3 for keys, 2 for values in TurboQuant)
        seed: random seed for rotation matrix (fixed per model for consistency)
    """

    def __init__(self, bits: int = 3, seed: int = 42):
        self.bits = bits
        self.seed = seed
        self._sign_cache: dict[int, torch.Tensor] = {}

    def _get_signs(self, d: int, device) -> torch.Tensor:
        key = (d, str(device))
        if key not in self._sign_cache:
            g = torch.Generator()
            g.manual_seed(self.seed)
            signs = (torch.randint(0, 2, (d,), generator=g).float() * 2 - 1).to(device)
            self._sign_cache[key] = signs
        return self._sign_cache[key]

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        d = x.shape[-1]
        d2 = _next_pow2(d)
        if d2 != d:
            pad = torch.zeros(*x.shape[:-1], d2 - d, dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=-1)
        return x, d

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random Hadamard rotation to last dimension."""
        x, _ = self._pad(x)
        d = x.shape[-1]
        signs = self._get_signs(d, x.device)
        x = x * signs
        x = _hadamard_transform(x, normalize=True)
        return x

    def unrotate(self, x: torch.Tensor, orig_d: int) -> torch.Tensor:
        """Inverse rotation (Hadamard is self-inverse up to scaling)."""
        d = x.shape[-1]
        x = _hadamard_transform(x, normalize=True)  # H^T = H for normalized WHT
        signs = self._get_signs(d, x.device)
        x = x * signs
        return x[..., :orig_d]

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Quantize tensor x using PolarQuant.
        Returns (quantized codes, scale, original_dim).
        """
        orig_d = x.shape[-1]
        x_rot = self.rotate(x)

        n_levels = 2 ** self.bits
        # Per-tensor symmetric scalar quantization on rotated values
        max_val = x_rot.abs().max()
        scale = max_val / (n_levels // 2 - 1) if max_val > 0 else torch.tensor(1.0)
        q = (x_rot / scale).round().clamp(-(n_levels // 2), n_levels // 2 - 1)
        return q, scale, orig_d

    def dequantize(self, q: torch.Tensor, scale, orig_d: int) -> torch.Tensor:
        """Dequantize and un-rotate."""
        x_rot = q * scale
        return self.unrotate(x_rot, orig_d)

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """Full round-trip for simulation (no actual compression)."""
        q, scale, orig_d = self.quantize(x)
        return self.dequantize(q, scale, orig_d)


# Module-level default instances (key=3bit, value=2bit)
_KEY_QUANT = PolarQuant(bits=3, seed=42)
_VAL_QUANT = PolarQuant(bits=2, seed=137)


def polar_quant_key(x: torch.Tensor) -> torch.Tensor:
    """3-bit PolarQuant for KV cache keys."""
    return _KEY_QUANT.quantize_dequantize(x)


def polar_quant_value(x: torch.Tensor) -> torch.Tensor:
    """2-bit PolarQuant for KV cache values."""
    return _VAL_QUANT.quantize_dequantize(x)
