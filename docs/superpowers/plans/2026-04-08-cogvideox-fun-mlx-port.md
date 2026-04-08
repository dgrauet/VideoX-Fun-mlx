# CogVideoX-Fun MLX Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port CogVideoX-Fun (VAE + Transformer + Inpaint Pipeline) from PyTorch/diffusers to MLX, enabling VOID model inference on Apple Silicon.

**Architecture:** The port replaces PyTorch ops with MLX equivalents, leveraging `mlx-ops` (at `/Users/dgrauet/Work/mlx-ops/`) for reusable building blocks. The original code uses channels-first (NCDHW) tensors throughout; our MLX port uses channels-last (NDHWC) as MLX requires. We keep the same class names and config format so HuggingFace safetensors weights load directly. No diffusers dependency — all model code is self-contained MLX.

**Tech Stack:** MLX 0.27+, mlx-ops, safetensors, numpy (for positional embeddings), einops (optional)

---

## File Structure

All new MLX files go in `videox_fun_mlx/` — a new package parallel to the original `videox_fun/`. The original code stays untouched for reference.

```
videox_fun_mlx/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── cogvideox_vae.py          # VAE: CausalConv3d, Encoder3D, Decoder3D, AutoencoderKL
│   ├── cogvideox_transformer3d.py # Transformer: PatchEmbed, Block, Transformer3DModel
│   └── embeddings.py             # Positional embeddings: 3D sincos, timestep, RoPE helpers
├── pipeline/
│   ├── __init__.py
│   ├── pipeline_cogvideox_fun_inpaint.py  # Inpaint pipeline
│   └── scheduler.py              # DDIM scheduler (minimal)
└── utils.py                      # DiagonalGaussianDistribution, weight loading
```

Test files:

```
tests/
├── test_vae_layers.py       # Task 1-2: CausalConv3d, ResnetBlock, Up/Down blocks
├── test_vae_encode_decode.py # Task 3: Full VAE encode/decode
├── test_transformer.py      # Task 4-5: PatchEmbed, Block, full transformer
├── test_pipeline.py         # Task 7: End-to-end pipeline
```

**Key design decisions:**
- **Channels-last everywhere.** MLX Conv3d expects NDHWC. We convert weights on load (NCDHW->NDHWC) and keep data in NDHWC throughout. The original PyTorch code is NCDHW.
- **No diffusers dependency.** We reimplement the ~5 diffusers classes we need (ConfigMixin pattern replaced by simple JSON config + `from_pretrained`). This keeps the MLX port clean and installable without PyTorch.
- **Conv cache for causal convolutions.** The original VAE passes `conv_cache` dicts through every layer for temporal causality. We preserve this pattern exactly — it's critical for temporal coherence in frame-by-frame decoding.
- **Weight compatibility.** Class and parameter names match the original so `safetensors` weights load via key remapping (only transpose conv weights, no renaming needed).

---

### Task 1: VAE Foundation Layers

**Files:**
- Create: `videox_fun_mlx/__init__.py`
- Create: `videox_fun_mlx/models/__init__.py`
- Create: `videox_fun_mlx/models/cogvideox_vae.py` (first 400 lines: CausalConv3d, SpatialNorm3D, Upsample3D, ResnetBlock3D)
- Create: `videox_fun_mlx/utils.py`
- Test: `tests/test_vae_layers.py`

This task ports the foundational layers the VAE is built from. Every layer in the VAE depends on these.

**Source reference:** `videox_fun/models/cogvideox_vae.py:40-404`

- [ ] **Step 1: Create package structure**

```bash
mkdir -p videox_fun_mlx/models videox_fun_mlx/pipeline tests
touch videox_fun_mlx/__init__.py videox_fun_mlx/models/__init__.py videox_fun_mlx/pipeline/__init__.py
```

- [ ] **Step 2: Write failing test for CogVideoXCausalConv3d**

```python
# tests/test_vae_layers.py
import mlx.core as mx
from videox_fun_mlx.models.cogvideox_vae import CogVideoXCausalConv3d

class TestCausalConv3d:
    def test_output_shape_constant_pad(self):
        """Causal conv with constant padding preserves temporal dim, returns conv_cache."""
        conv = CogVideoXCausalConv3d(4, 8, kernel_size=3, pad_mode="constant")
        # MLX channels-last: (B, D, H, W, C)
        x = mx.random.normal((1, 4, 8, 8, 4))
        out, cache = conv(x)
        mx.eval(out, cache)
        assert out.shape == (1, 4, 8, 8, 8)
        assert cache is not None

    def test_output_shape_replicate_pad(self):
        conv = CogVideoXCausalConv3d(4, 8, kernel_size=3, pad_mode="replicate")
        x = mx.random.normal((1, 4, 8, 8, 4))
        out, cache = conv(x)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 8)
        assert cache is None  # replicate mode has no cache

    def test_conv_cache_continuity(self):
        """Feeding cache from first call to second should produce same result as full sequence."""
        conv = CogVideoXCausalConv3d(4, 8, kernel_size=3, pad_mode="constant")
        x_full = mx.random.normal((1, 8, 6, 6, 4))
        out_full, _ = conv(x_full)
        mx.eval(out_full)

        x1 = x_full[:, :4]
        x2 = x_full[:, 4:]
        out1, cache = conv(x1)
        out2, _ = conv(x2, conv_cache=cache)
        out_split = mx.concatenate([out1, out2], axis=1)
        mx.eval(out_split)
        assert mx.allclose(out_full, out_split, atol=1e-4)
```

Run: `cd /Users/dgrauet/Work/VideoX-Fun-mlx && python3 -m pytest tests/test_vae_layers.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement CogVideoXCausalConv3d**

Port `videox_fun/models/cogvideox_vae.py:40-148`. Key changes from PyTorch:
- Replace `nn.Conv3d` with `mlx.nn.Conv3d` (channels-last: NDHWC)
- Replace `F.pad(mode="replicate")` with manual replication (pad first temporal frame)
- Replace `F.pad(mode="constant")` with `mx.pad`
- `CogVideoXSafeConv3d` memory chunking: implement by splitting along temporal dim and concatenating (same logic, MLX ops)
- Conv cache slicing: `inputs[:, :, -k+1:]` becomes `inputs[:, -k+1:]` (temporal is dim 1 in NDHWC)

The implementation must return `(output, conv_cache)` tuple from `__call__`, matching the original's `forward` signature.

```python
# videox_fun_mlx/models/cogvideox_vae.py

from typing import Dict, Optional, Tuple, Union
import mlx.core as mx
import mlx.nn as nn


class CogVideoXCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "constant",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        self.pad_mode = pad_mode
        self.height_pad = (height_kernel_size - 1) // 2
        self.width_pad = (width_kernel_size - 1) // 2
        self.time_pad = time_kernel_size - 1
        self.time_kernel_size = time_kernel_size

        stride = stride if isinstance(stride, tuple) else (stride, 1, 1)
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=True,
        )

    def __call__(
        self, inputs: mx.array, conv_cache: Optional[mx.array] = None
    ) -> Tuple[mx.array, Optional[mx.array]]:
        # inputs: (B, D, H, W, C) -- channels-last
        new_cache = None

        if self.pad_mode == "replicate":
            if self.time_pad > 0:
                first = mx.repeat(inputs[:, :1], self.time_pad, axis=1)
                inputs = mx.concatenate([first, inputs], axis=1)
            if self.height_pad > 0 or self.width_pad > 0:
                inputs = mx.pad(inputs, [
                    (0, 0), (0, 0),
                    (self.height_pad, self.height_pad),
                    (self.width_pad, self.width_pad),
                    (0, 0),
                ])
        else:
            if self.time_kernel_size > 1:
                if conv_cache is not None:
                    cached = [conv_cache]
                else:
                    cached = [mx.repeat(inputs[:, :1], self.time_pad, axis=1)]
                inputs = mx.concatenate(cached + [inputs], axis=1)
            new_cache = inputs[:, -self.time_kernel_size + 1:] if self.time_kernel_size > 1 else None
            if self.height_pad > 0 or self.width_pad > 0:
                inputs = mx.pad(inputs, [
                    (0, 0), (0, 0),
                    (self.height_pad, self.height_pad),
                    (self.width_pad, self.width_pad),
                    (0, 0),
                ])

        output = self.conv(inputs)
        return output, new_cache
```

- [ ] **Step 4: Run tests to verify CausalConv3d passes**

Run: `python3 -m pytest tests/test_vae_layers.py::TestCausalConv3d -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Write failing test for CogVideoXResnetBlock3D**

```python
# Append to tests/test_vae_layers.py
from videox_fun_mlx.models.cogvideox_vae import CogVideoXResnetBlock3D

class TestResnetBlock3D:
    def test_same_channels(self):
        block = CogVideoXResnetBlock3D(in_channels=32, out_channels=32)
        x = mx.random.normal((1, 4, 8, 8, 32))  # NDHWC
        out, cache = block(x)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 32)

    def test_channel_change(self):
        block = CogVideoXResnetBlock3D(in_channels=32, out_channels=64)
        x = mx.random.normal((1, 4, 8, 8, 32))
        out, cache = block(x)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 64)

    def test_with_temb(self):
        block = CogVideoXResnetBlock3D(in_channels=32, temb_channels=128)
        x = mx.random.normal((1, 4, 8, 8, 32))
        temb = mx.random.normal((1, 128))
        out, cache = block(x, temb=temb)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 32)
```

- [ ] **Step 6: Implement CogVideoXResnetBlock3D, CogVideoXSpatialNorm3D, CogVideoXUpsample3D**

Port `videox_fun/models/cogvideox_vae.py:150-273` and `276-404`. Key changes:
- **GroupNorm**: Use `mlx.nn.GroupNorm(... pytorch_compatible=True)`. The original uses channels-first GroupNorm; MLX GroupNorm operates on last dim by default which matches our NDHWC layout.
- **SpatialNorm3D**: Uses `F.interpolate` for resizing `zq` to match `f`. Replace with a helper that does nearest-neighbor interpolation per dimension. For the first-frame splitting pattern (lines 183-189), implement the same logic with MLX slicing.
- **Upsample3D**: The original uses `F.interpolate(scale_factor=2.0)`. For spatial-only upsampling, reshape to (B*D, H, W, C), use `mlx_ops.spatial.upsample_nearest`, reshape back. For compress_time mode (spatial+temporal), handle the first-frame special case.
- **ResnetBlock3D**: Straightforward port. Replace `get_activation("swish")` with `nn.silu`. Conv cache dict must be threaded through. The `temb` broadcast `[:, :, None, None, None]` becomes `reshape(B, 1, 1, 1, C)` for NDHWC.

- [ ] **Step 7: Run tests**

Run: `python3 -m pytest tests/test_vae_layers.py -v`
Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
git add videox_fun_mlx/ tests/test_vae_layers.py
git commit -m "feat: port VAE foundation layers (CausalConv3d, ResnetBlock3D, Upsample3D)"
```

---

### Task 2: VAE Encoder and Decoder

**Files:**
- Modify: `videox_fun_mlx/models/cogvideox_vae.py` (add DownBlock3D, MidBlock3D, UpBlock3D, Encoder3D, Decoder3D)
- Test: `tests/test_vae_layers.py` (append)

**Source reference:** `videox_fun/models/cogvideox_vae.py:407-1066`

- [ ] **Step 1: Write failing test for Encoder3D and Decoder3D**

```python
# Append to tests/test_vae_layers.py
from videox_fun_mlx.models.cogvideox_vae import CogVideoXEncoder3D, CogVideoXDecoder3D

class TestEncoder3D:
    def test_output_shape(self):
        enc = CogVideoXEncoder3D(
            in_channels=3, out_channels=16,
            block_out_channels=(32, 64, 64, 128),
            layers_per_block=1,
        )
        x = mx.random.normal((1, 8, 32, 32, 3))
        out, cache = enc(x)
        mx.eval(out)
        assert out.shape[0] == 1
        assert out.shape[-1] == 32  # 2 * latent_channels

class TestDecoder3D:
    def test_output_shape(self):
        dec = CogVideoXDecoder3D(
            in_channels=16, out_channels=3,
            block_out_channels=(32, 64, 64, 128),
            layers_per_block=1,
        )
        z = mx.random.normal((1, 2, 4, 4, 16))
        out, cache = dec(z)
        mx.eval(out)
        assert out.shape[0] == 1
        assert out.shape[-1] == 3
```

- [ ] **Step 2: Implement DownBlock3D, MidBlock3D, UpBlock3D**

Port `videox_fun/models/cogvideox_vae.py:407-740`. These are containers of ResnetBlock3D + optional downsamplers/upsamplers. Key changes:
- Remove gradient checkpointing code (inference only)
- `CogVideoXDownsample3D` from diffusers: implement as Conv3d with stride (2,2,2) or (1,2,2) depending on `compress_time`
- Thread `conv_cache` through each block

- [ ] **Step 3: Implement Encoder3D and Decoder3D**

Port `videox_fun/models/cogvideox_vae.py:743-1066`. Same pattern: container of down/mid/up blocks. Key notes:
- Encoder output has `2 * out_channels` (mean + logvar for VAE)
- Decoder uses `CogVideoXSpatialNorm3D` (spatial conditioning from latent)
- `temporal_compress_level = int(np.log2(temporal_compression_ratio))` determines which blocks also compress time

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_vae_layers.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add videox_fun_mlx/models/cogvideox_vae.py tests/test_vae_layers.py
git commit -m "feat: port VAE Encoder3D and Decoder3D"
```

---

### Task 3: Full AutoencoderKL + Weight Loading

**Files:**
- Modify: `videox_fun_mlx/models/cogvideox_vae.py` (add AutoencoderKLCogVideoX)
- Create: `videox_fun_mlx/utils.py` (DiagonalGaussianDistribution, weight loading)
- Test: `tests/test_vae_encode_decode.py`

**Source reference:** `videox_fun/models/cogvideox_vae.py:1069-1674`

- [ ] **Step 1: Write failing test for DiagonalGaussianDistribution**

```python
# tests/test_vae_encode_decode.py
import mlx.core as mx
from videox_fun_mlx.utils import DiagonalGaussianDistribution

class TestDiagonalGaussian:
    def test_sample_shape(self):
        params = mx.random.normal((1, 4, 8, 8, 32))  # 16 mean + 16 logvar
        dist = DiagonalGaussianDistribution(params)
        sample = dist.sample()
        mx.eval(sample)
        assert sample.shape == (1, 4, 8, 8, 16)

    def test_mode(self):
        params = mx.random.normal((1, 4, 8, 8, 32))
        dist = DiagonalGaussianDistribution(params)
        mode = dist.mode()
        mx.eval(mode)
        assert mode.shape == (1, 4, 8, 8, 16)
```

- [ ] **Step 2: Implement DiagonalGaussianDistribution**

```python
# videox_fun_mlx/utils.py
import mlx.core as mx

class DiagonalGaussianDistribution:
    def __init__(self, parameters: mx.array):
        self.mean, self.logvar = mx.split(parameters, 2, axis=-1)
        self.logvar = mx.clip(self.logvar, -30.0, 20.0)
        self.std = mx.exp(0.5 * self.logvar)

    def sample(self) -> mx.array:
        return self.mean + self.std * mx.random.normal(self.mean.shape)

    def mode(self) -> mx.array:
        return self.mean
```

- [ ] **Step 3: Implement AutoencoderKLCogVideoX**

Port `videox_fun/models/cogvideox_vae.py:1069-1674`. Key pieces:
- `__init__`: Create Encoder3D + Decoder3D + optional quant_conv
- `encode`: Run encoder, return `DiagonalGaussianDistribution`
- `decode`: Run decoder, return output tensor
- `from_pretrained`: Load config.json, create model, load safetensors weights with conv weight transposition
- Frame batching in `_encode`/`_decode`: keep the temporal slicing logic
- Skip tiling for now (mark as TODO)

- [ ] **Step 4: Write smoke test for AutoencoderKL**

```python
class TestAutoencoderKL:
    def test_from_config(self):
        from videox_fun_mlx.models.cogvideox_vae import AutoencoderKLCogVideoX
        model = AutoencoderKLCogVideoX(
            in_channels=3, out_channels=3,
            block_out_channels=(32, 64),
            latent_channels=4,
            layers_per_block=1,
        )
        x = mx.random.normal((1, 4, 16, 16, 3))
        posterior = model.encode(x)
        z = posterior.mode()
        mx.eval(z)
        assert z.shape[-1] == 4
```

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/test_vae_encode_decode.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add videox_fun_mlx/utils.py videox_fun_mlx/models/cogvideox_vae.py tests/test_vae_encode_decode.py
git commit -m "feat: port AutoencoderKLCogVideoX with weight loading"
```

---

### Task 4: Positional Embeddings and Helpers

**Files:**
- Create: `videox_fun_mlx/models/embeddings.py`
- Test: `tests/test_transformer.py`

**Source reference:** `videox_fun/models/cogvideox_transformer3d.py:116-244` (PatchEmbed), `videox_fun/pipeline/pipeline_cogvideox_fun_inpaint.py:49-140` (3D RoPE)

- [ ] **Step 1: Write failing test for 3D sincos positional embeddings**

```python
# tests/test_transformer.py
import mlx.core as mx
from videox_fun_mlx.models.embeddings import get_3d_sincos_pos_embed

class TestPositionalEmbeddings:
    def test_3d_sincos_shape(self):
        embed = get_3d_sincos_pos_embed(
            embed_dim=256,
            spatial_size=(8, 8),
            temporal_size=4,
        )
        mx.eval(embed)
        assert embed.shape == (256, 256)
```

- [ ] **Step 2: Implement get_3d_sincos_pos_embed**

Use numpy for grid math (one-time computation), convert to `mx.array`. Port from diffusers `get_3d_sincos_pos_embed`.

- [ ] **Step 3: Implement get_3d_rotary_pos_embed and apply_rotary_emb**

Port `videox_fun/pipeline/pipeline_cogvideox_fun_inpaint.py:49-140`. Returns `(cos, sin)` tuple.

```python
def apply_rotary_emb(x: mx.array, freqs: tuple) -> mx.array:
    cos, sin = freqs
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_transformer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add videox_fun_mlx/models/embeddings.py tests/test_transformer.py
git commit -m "feat: port positional embeddings (3D sincos, 3D RoPE)"
```

---

### Task 5: Transformer Model

**Files:**
- Create: `videox_fun_mlx/models/cogvideox_transformer3d.py`
- Modify: `tests/test_transformer.py` (append)

**Source reference:** `videox_fun/models/cogvideox_transformer3d.py:44-916`

- [ ] **Step 1: Write failing test for CogVideoXBlock**

```python
from videox_fun_mlx.models.cogvideox_transformer3d import CogVideoXBlock

class TestCogVideoXBlock:
    def test_output_shapes(self):
        block = CogVideoXBlock(
            dim=256, num_attention_heads=4,
            attention_head_dim=64, time_embed_dim=256,
        )
        hidden = mx.random.normal((1, 64, 256))
        encoder = mx.random.normal((1, 16, 256))
        temb = mx.random.normal((1, 256))
        out_h, out_e = block(hidden, encoder, temb)
        mx.eval(out_h, out_e)
        assert out_h.shape == (1, 64, 256)
        assert out_e.shape == (1, 16, 256)
```

- [ ] **Step 2: Implement CogVideoXLayerNormZero**

Modulated LayerNorm producing separate shift/scale/gate for video and text streams:

```python
class CogVideoXLayerNormZero(nn.Module):
    def __init__(self, conditioning_dim, embedding_dim, elementwise_affine=True, eps=1e-5, bias=True):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, affine=elementwise_affine)
        self.linear = nn.Linear(conditioning_dim, 6 * embedding_dim, bias=bias)
        self.norm_enc = nn.LayerNorm(embedding_dim, eps=eps, affine=elementwise_affine)

    def __call__(self, hidden_states, encoder_hidden_states, temb):
        modulation = nn.silu(temb)
        modulation = self.linear(modulation)
        if modulation.ndim == 2:
            modulation = mx.expand_dims(modulation, 1)
        shift_h, scale_h, gate_h, shift_e, scale_e, gate_e = mx.split(modulation, 6, axis=-1)
        hidden_states = self.norm(hidden_states) * (1 + scale_h) + shift_h
        encoder_hidden_states = self.norm_enc(encoder_hidden_states) * (1 + scale_e) + shift_e
        return hidden_states, encoder_hidden_states, gate_h, gate_e
```

- [ ] **Step 3: Implement CogVideoXBlock (attention + FFN)**

Attention: concatenate text + video tokens, project Q/K/V, apply RoPE to video portion only, run `mx.fast.scaled_dot_product_attention`, split back. FFN: Linear -> GELU -> Linear.

- [ ] **Step 4: Implement CogVideoXPatchEmbed**

Port `videox_fun/models/cogvideox_transformer3d.py:116-244`. Handle both patch_size_t=None (v1.0 Conv2d) and patch_size_t!=None (v1.5 Linear) paths. Replace `F.interpolate(mode='trilinear')` for pos embedding resize with nearest-neighbor interpolation.

- [ ] **Step 5: Implement CogVideoXTransformer3DModel**

Port lines 366-757. Orchestrates: time embedding -> patch embedding -> N transformer blocks -> final norm -> unpatchify. Skip multi-GPU and gradient checkpointing.

- [ ] **Step 6: Implement from_pretrained**

Load config.json, instantiate, load safetensors with weight remapping. Handle `patch_embed.proj.weight` size mismatch for inpainting models.

- [ ] **Step 7: Write integration test**

```python
from videox_fun_mlx.models.cogvideox_transformer3d import CogVideoXTransformer3DModel

class TestTransformer3DModel:
    def test_forward_small(self):
        model = CogVideoXTransformer3DModel(
            num_attention_heads=4, attention_head_dim=32,
            in_channels=16, out_channels=16,
            time_embed_dim=128, text_embed_dim=256,
            num_layers=2, sample_width=8, sample_height=8,
            sample_frames=5, patch_size=2,
            temporal_compression_ratio=4,
            max_text_seq_length=16,
            use_rotary_positional_embeddings=True,
        )
        # Transformer expects (B, F, C, H, W) for input, converts internally
        hidden = mx.random.normal((1, 1, 16, 8, 8))
        encoder = mx.random.normal((1, 16, 256))
        timestep = mx.array([500.0])
        out = model(hidden, encoder, timestep)
        mx.eval(out)
        assert out.shape[0] == 1
```

- [ ] **Step 8: Run tests**

Run: `python3 -m pytest tests/test_transformer.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add videox_fun_mlx/models/cogvideox_transformer3d.py tests/test_transformer.py
git commit -m "feat: port CogVideoX Transformer3DModel"
```

---

### Task 6: Scheduler

**Files:**
- Create: `videox_fun_mlx/pipeline/scheduler.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_pipeline.py
import mlx.core as mx
from videox_fun_mlx.pipeline.scheduler import DDIMScheduler

class TestDDIMScheduler:
    def test_timesteps(self):
        sched = DDIMScheduler(num_train_timesteps=1000, num_inference_steps=50)
        assert len(sched.timesteps) == 50

    def test_step(self):
        sched = DDIMScheduler(num_train_timesteps=1000, num_inference_steps=50)
        sample = mx.random.normal((1, 4, 4, 4, 16))
        noise_pred = mx.random.normal((1, 4, 4, 4, 16))
        out = sched.step(noise_pred, sched.timesteps[0], sample)
        mx.eval(out)
        assert out.shape == sample.shape
```

- [ ] **Step 2: Implement DDIMScheduler**

Minimal DDIM: `alphas_cumprod`, `set_timesteps`, `step`, `add_noise`. ~80 lines. Standard DDIM math.

- [ ] **Step 3: Run tests, commit**

```bash
python3 -m pytest tests/test_pipeline.py::TestDDIMScheduler -v
git add videox_fun_mlx/pipeline/scheduler.py tests/test_pipeline.py
git commit -m "feat: implement DDIM scheduler for MLX"
```

---

### Task 7: Inpaint Pipeline

**Files:**
- Create: `videox_fun_mlx/pipeline/pipeline_cogvideox_fun_inpaint.py`
- Modify: `tests/test_pipeline.py` (append)

**Source reference:** `videox_fun/pipeline/pipeline_cogvideox_fun_inpaint.py:287-1134`

- [ ] **Step 1: Implement pipeline**

Accepts pre-computed `prompt_embeds` (skip T5 for now). Core loop:
1. Encode video to latents via VAE
2. Encode mask to latent space
3. Add noise
4. Compute RoPE for current resolution
5. Denoising loop with scheduler
6. Decode latents via VAE

- [ ] **Step 2: Write integration test with tiny random model**

```python
class TestInpaintPipelineIntegration:
    def test_forward_random_weights(self):
        from videox_fun_mlx.models.cogvideox_vae import AutoencoderKLCogVideoX
        from videox_fun_mlx.models.cogvideox_transformer3d import CogVideoXTransformer3DModel
        from videox_fun_mlx.pipeline.scheduler import DDIMScheduler
        from videox_fun_mlx.pipeline.pipeline_cogvideox_fun_inpaint import CogVideoXFunInpaintPipeline

        vae = AutoencoderKLCogVideoX(
            in_channels=3, out_channels=3,
            block_out_channels=(32, 64),
            latent_channels=4, layers_per_block=1,
        )
        transformer = CogVideoXTransformer3DModel(
            num_attention_heads=4, attention_head_dim=16,
            in_channels=12, out_channels=4,
            time_embed_dim=64, text_embed_dim=128,
            num_layers=1, sample_width=4, sample_height=4,
            sample_frames=5, patch_size=2,
            temporal_compression_ratio=4,
            max_text_seq_length=8,
            use_rotary_positional_embeddings=True,
        )
        scheduler = DDIMScheduler(num_inference_steps=2)
        pipe = CogVideoXFunInpaintPipeline(
            vae=vae, transformer=transformer, scheduler=scheduler,
        )
        prompt_embeds = mx.random.normal((1, 8, 128))
        video = mx.random.normal((1, 4, 16, 16, 3))
        mask = mx.zeros((1, 4, 16, 16, 1))
        out = pipe(prompt_embeds=prompt_embeds, video=video, mask=mask, num_inference_steps=2)
        mx.eval(out)
        assert out.shape[0] == 1
```

- [ ] **Step 3: Run tests, commit**

```bash
python3 -m pytest tests/test_pipeline.py -v
git add videox_fun_mlx/pipeline/ tests/test_pipeline.py
git commit -m "feat: port CogVideoX-Fun inpaint pipeline"
```

---

### Task 8: Real Weight Validation

**Files:**
- Create: `scripts/test_with_real_weights.py`

Manual validation requiring actual CogVideoX-Fun-5B weights.

- [ ] **Step 1: Write validation script**

```python
# scripts/test_with_real_weights.py
"""
Usage: python scripts/test_with_real_weights.py --model-path /path/to/CogVideoX-Fun-V1.5-5b-InP
"""
import argparse
import mlx.core as mx
from videox_fun_mlx.models.cogvideox_vae import AutoencoderKLCogVideoX
from videox_fun_mlx.models.cogvideox_transformer3d import CogVideoXTransformer3DModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    print("Loading VAE...")
    vae = AutoencoderKLCogVideoX.from_pretrained(f"{args.model_path}/vae")

    print("Testing VAE encode/decode...")
    x = mx.random.normal((1, 5, 64, 64, 3))
    z = vae.encode(x).mode()
    mx.eval(z)
    print(f"  Latent shape: {z.shape}")

    recon = vae.decode(z)
    mx.eval(recon)
    print(f"  Reconstruction shape: {recon.shape}")

    print("Loading Transformer...")
    transformer = CogVideoXTransformer3DModel.from_pretrained(f"{args.model_path}/transformer")
    param_count = sum(p.size for _, p in transformer.parameters().items())
    print(f"  Parameters: {param_count / 1e6:.1f}M")

    print("All checks passed!")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run with real weights, commit**

```bash
python3 scripts/test_with_real_weights.py --model-path /path/to/CogVideoX-Fun-V1.5-5b-InP
git add scripts/test_with_real_weights.py
git commit -m "feat: add real-weight validation script"
```

---

## Appendix: Key Mapping Reference

### Tensor Layout Conversion (PyTorch to MLX)

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `(B, C, D, H, W)` | `(B, D, H, W, C)` | All video tensors |
| `(B, C, H, W)` | `(B, H, W, C)` | 2D image tensors |
| Conv3d weight `(O, I, kD, kH, kW)` | `(O, kD, kH, kW, I)` | Use `mlx_ops.layout.convert_conv_weights` |
| Conv2d weight `(O, I, kH, kW)` | `(O, kH, kW, I)` | Same helper |

### PyTorch Op to MLX Equivalent

| PyTorch | MLX |
|---------|-----|
| `F.scaled_dot_product_attention` | `mx.fast.scaled_dot_product_attention` |
| `F.pad(mode="replicate")` | Manual: replicate edge slices + concatenate |
| `F.pad(mode="constant")` | `mx.pad(...)` |
| `F.interpolate(mode="nearest", scale_factor=2)` | `mlx_ops.spatial.upsample_nearest` or reshape+repeat |
| `F.interpolate(mode="trilinear")` | Decompose: spatial bilinear + temporal linear, or nearest approx |
| `nn.GroupNorm` | `mlx.nn.GroupNorm(pytorch_compatible=True)` |
| `nn.SiLU` / `F.silu` | `mlx.nn.silu` |
| `torch.cat` | `mx.concatenate` |
| `tensor.view/reshape` | `mx.reshape` |
| `tensor.permute` | `mx.transpose` |
| `tensor.split` | `mx.split` |
| `tensor.chunk` | `mx.split` with equal parts |
| `torch.zeros_like` | `mx.zeros_like` |
| `einops.rearrange` | `einops.rearrange` (MLX backend works natively) |
