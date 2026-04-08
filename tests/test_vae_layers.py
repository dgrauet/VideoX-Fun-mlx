"""Tests for VAE foundation layers: CausalConv3d, ResnetBlock3D, SpatialNorm3D, Upsample3D."""

import mlx.core as mx
import pytest

from videox_fun_mlx.models.cogvideox_vae import (
    CogVideoXCausalConv3d,
    CogVideoXResnetBlock3D,
    CogVideoXSpatialNorm3D,
    CogVideoXUpsample3D,
)


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
        """Replicate pad mode preserves shape and returns no cache."""
        conv = CogVideoXCausalConv3d(4, 8, kernel_size=3, pad_mode="replicate")
        x = mx.random.normal((1, 4, 8, 8, 4))
        out, cache = conv(x)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 8)
        assert cache is None

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

    def test_kernel_size_1(self):
        """Kernel size 1 should have no temporal padding and no cache."""
        conv = CogVideoXCausalConv3d(4, 8, kernel_size=1, pad_mode="constant")
        x = mx.random.normal((1, 4, 8, 8, 4))
        out, cache = conv(x)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 8)
        assert cache is None

    def test_stride(self):
        """Temporal stride should reduce output temporal dimension."""
        conv = CogVideoXCausalConv3d(4, 8, kernel_size=3, stride=2, pad_mode="constant")
        x = mx.random.normal((1, 8, 8, 8, 4))
        out, cache = conv(x)
        mx.eval(out)
        # Stride 2 on temporal dim: (8 + 2 pad - 3) / 2 + 1 = 4 (approx)
        assert out.shape[0] == 1
        assert out.shape[2] == 8  # spatial preserved
        assert out.shape[3] == 8
        assert out.shape[4] == 8  # out_channels


class TestSpatialNorm3D:
    def test_output_shape(self):
        """SpatialNorm3D should output same shape as input f."""
        norm = CogVideoXSpatialNorm3D(f_channels=32, zq_channels=16, groups=8)
        f = mx.random.normal((1, 4, 8, 8, 32))
        zq = mx.random.normal((1, 4, 8, 8, 16))
        out, cache = norm(f, zq)
        mx.eval(out)
        assert out.shape == f.shape

    def test_with_different_zq_shape(self):
        """SpatialNorm3D should handle zq with different spatial dims via interpolation."""
        norm = CogVideoXSpatialNorm3D(f_channels=32, zq_channels=16, groups=8)
        f = mx.random.normal((1, 4, 8, 8, 32))
        zq = mx.random.normal((1, 2, 4, 4, 16))  # smaller than f
        out, cache = norm(f, zq)
        mx.eval(out)
        assert out.shape == f.shape

    def test_odd_temporal_dim(self):
        """SpatialNorm3D should handle odd temporal dimensions (first-frame splitting)."""
        norm = CogVideoXSpatialNorm3D(f_channels=32, zq_channels=16, groups=8)
        f = mx.random.normal((1, 5, 8, 8, 32))  # odd D=5
        zq = mx.random.normal((1, 3, 4, 4, 16))
        out, cache = norm(f, zq)
        mx.eval(out)
        assert out.shape == f.shape


class TestUpsample3D:
    def test_spatial_only(self):
        """Upsample3D without compress_time should 2x spatial dims only."""
        up = CogVideoXUpsample3D(in_channels=16, out_channels=16, compress_time=False)
        x = mx.random.normal((1, 4, 8, 8, 16))
        out = up(x)
        mx.eval(out)
        assert out.shape == (1, 4, 16, 16, 16)

    def test_compress_time_even(self):
        """Upsample3D with compress_time and even D should 2x all dims."""
        up = CogVideoXUpsample3D(in_channels=16, out_channels=16, compress_time=True)
        x = mx.random.normal((1, 4, 8, 8, 16))
        out = up(x)
        mx.eval(out)
        assert out.shape == (1, 8, 16, 16, 16)

    def test_compress_time_odd(self):
        """Upsample3D with compress_time and odd D>1 should handle first-frame split."""
        up = CogVideoXUpsample3D(in_channels=16, out_channels=16, compress_time=True)
        x = mx.random.normal((1, 5, 8, 8, 16))
        out = up(x)
        mx.eval(out)
        # D=5 odd: first frame stays 1, rest 4 frames -> 8 frames. total = 1 + 8 = 9
        assert out.shape[0] == 1
        assert out.shape[1] == 9  # 1 + (5-1)*2
        assert out.shape[2] == 16
        assert out.shape[3] == 16
        assert out.shape[4] == 16

    def test_compress_time_single_frame(self):
        """Upsample3D with compress_time and D=1 should keep D=1 and 2x spatial."""
        up = CogVideoXUpsample3D(in_channels=16, out_channels=16, compress_time=True)
        x = mx.random.normal((1, 1, 8, 8, 16))
        out = up(x)
        mx.eval(out)
        assert out.shape == (1, 1, 16, 16, 16)

    def test_channel_change(self):
        """Upsample3D should change channel count via the 2D conv."""
        up = CogVideoXUpsample3D(in_channels=16, out_channels=32, compress_time=False)
        x = mx.random.normal((1, 4, 8, 8, 16))
        out = up(x)
        mx.eval(out)
        assert out.shape == (1, 4, 16, 16, 32)


class TestResnetBlock3D:
    def test_same_channels(self):
        """ResnetBlock3D with same in/out channels."""
        block = CogVideoXResnetBlock3D(in_channels=32, out_channels=32)
        x = mx.random.normal((1, 4, 8, 8, 32))
        out, cache = block(x)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 32)

    def test_channel_change(self):
        """ResnetBlock3D with different in/out channels uses shortcut."""
        block = CogVideoXResnetBlock3D(in_channels=32, out_channels=64)
        x = mx.random.normal((1, 4, 8, 8, 32))
        out, cache = block(x)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 64)

    def test_with_temb(self):
        """ResnetBlock3D with time embedding."""
        block = CogVideoXResnetBlock3D(in_channels=32, temb_channels=128)
        x = mx.random.normal((1, 4, 8, 8, 32))
        temb = mx.random.normal((1, 128))
        out, cache = block(x, temb=temb)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 32)

    def test_conv_shortcut(self):
        """ResnetBlock3D with conv_shortcut=True for channel change."""
        block = CogVideoXResnetBlock3D(
            in_channels=32, out_channels=64, conv_shortcut=True
        )
        x = mx.random.normal((1, 4, 8, 8, 32))
        out, cache = block(x)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 64)

    def test_conv_cache_returned(self):
        """ResnetBlock3D should return conv_cache dict with expected keys."""
        block = CogVideoXResnetBlock3D(in_channels=32, out_channels=32)
        x = mx.random.normal((1, 4, 8, 8, 32))
        out, cache = block(x)
        mx.eval(out)
        assert "conv1" in cache
        assert "conv2" in cache

    def test_no_temb_channels(self):
        """ResnetBlock3D with temb_channels=0 should work without temb_proj."""
        block = CogVideoXResnetBlock3D(in_channels=32, temb_channels=0)
        x = mx.random.normal((1, 4, 8, 8, 32))
        out, cache = block(x)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 32)
