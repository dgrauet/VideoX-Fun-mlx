"""Utility classes for CogVideoX-Fun MLX port."""

import json
import os
from pathlib import Path
from typing import Optional

import mlx.core as mx


class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution parameterized by mean and logvar.

    Used by the VAE to represent the latent distribution.
    """

    def __init__(self, parameters: mx.array):
        self.mean, self.logvar = mx.split(parameters, 2, axis=-1)
        self.logvar = mx.clip(self.logvar, -30.0, 20.0)
        self.std = mx.exp(0.5 * self.logvar)

    def sample(self) -> mx.array:
        return self.mean + self.std * mx.random.normal(self.mean.shape)

    def mode(self) -> mx.array:
        return self.mean


def load_config(path: str) -> dict:
    """Load model config from a JSON file."""
    config_file = os.path.join(path, "config.json")
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"{config_file} does not exist")
    with open(config_file, "r") as f:
        return json.load(f)


def load_weights(path: str) -> dict:
    """Load safetensors weights from a directory, handling sharded files."""
    from pathlib import Path

    p = Path(path)
    safetensors_files = sorted(p.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {path}")

    weights = {}
    for f in safetensors_files:
        w = mx.load(str(f))
        weights.update(w)
    return weights


def transpose_conv_weight(w: mx.array) -> mx.array:
    """Transpose a conv weight from PyTorch to MLX format.

    PyTorch Conv3d: (O, I, kD, kH, kW) -> MLX: (O, kD, kH, kW, I)
    PyTorch Conv2d: (O, I, kH, kW) -> MLX: (O, kH, kW, I)
    PyTorch Conv1d: (O, I, K) -> MLX: (O, K, I)
    """
    if w.ndim == 5:
        return w.transpose(0, 2, 3, 4, 1)
    elif w.ndim == 4:
        return w.transpose(0, 2, 3, 1)
    elif w.ndim == 3:
        return w.transpose(0, 2, 1)
    return w


def _is_conv_weight(key: str, weight: mx.array) -> bool:
    """Heuristic: a weight is a conv weight if it ends with .weight and has 3-5 dims
    and is not a linear/embedding (which are 2D)."""
    if not key.endswith(".weight"):
        return False
    return weight.ndim >= 3


def convert_pytorch_weights(weights: dict) -> dict:
    """Convert PyTorch weights to MLX format (transpose conv weights)."""
    converted = {}
    for key, w in weights.items():
        if _is_conv_weight(key, w):
            converted[key] = transpose_conv_weight(w)
        else:
            converted[key] = w
    return converted
