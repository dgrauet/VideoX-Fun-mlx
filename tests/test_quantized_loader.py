"""Quantized-weight loading robustness (VOID dogfood 2026-07-20).

quantize_model_from_weights silently returned when quantize_config.json was
missing even though the weights plainly carried .scales keys — the Linears
stayed unconverted and the run died 3 layers deep in a cryptic addmm shape
error. bits/group_size are fully determined by the shapes (for a Linear of
in_dim I: scales.shape[-1] = I/group_size, packed weight.shape[-1] =
I*bits/32), so infer them instead of bailing out.
"""

import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from videox_fun_mlx.utils import quantize_model_from_weights


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 64)

    def __call__(self, x):
        return self.proj(x)


def _quantized_weights(bits, group_size):
    ref = Tiny()
    nn.quantize(ref, group_size=group_size, bits=bits, class_predicate=lambda p, m: isinstance(m, nn.Linear))
    flat = dict(nn.utils.tree_flatten(ref.parameters()))
    return ref, {k: v for k, v in flat.items()}


@pytest.mark.parametrize("bits,group_size", [(4, 64), (8, 64), (4, 32)])
def test_infers_bits_and_group_size_without_config(tmp_path, bits, group_size):
    """No quantize_config.json on disk: infer from the weight shapes."""
    ref, weights = _quantized_weights(bits, group_size)

    model = Tiny()
    quantize_model_from_weights(model, weights, str(tmp_path), "transformer")

    assert isinstance(model.proj, nn.QuantizedLinear), (
        "Linear must be converted even without quantize_config.json"
    )
    assert model.proj.bits == bits
    assert model.proj.group_size == group_size

    model.load_weights(list(weights.items()), strict=False)
    x = mx.random.normal((2, 128))
    out = model(x)
    want = ref(x)
    assert np.allclose(np.array(out), np.array(want), atol=1e-5)


def test_config_file_still_wins(tmp_path):
    """Explicit quantize_config.json keeps taking precedence."""
    _, weights = _quantized_weights(4, 32)
    (tmp_path / "quantize_config.json").write_text(
        json.dumps({"quantization": {"bits": 4, "group_size": 32}})
    )
    model = Tiny()
    quantize_model_from_weights(model, weights, str(tmp_path), "transformer")
    assert isinstance(model.proj, nn.QuantizedLinear)
    assert model.proj.group_size == 32


def test_non_quantized_weights_are_untouched(tmp_path):
    model = Tiny()
    flat = dict(nn.utils.tree_flatten(Tiny().parameters()))
    quantize_model_from_weights(model, flat, str(tmp_path), "transformer")
    assert isinstance(model.proj, nn.Linear)
    assert not isinstance(model.proj, nn.QuantizedLinear)
