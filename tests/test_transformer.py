"""Tests for positional embeddings (Task 4) and transformer components (Task 5)."""

import mlx.core as mx
import pytest

from videox_fun_mlx.models.embeddings import (
    apply_rotary_emb,
    get_3d_rotary_pos_embed,
    get_3d_sincos_pos_embed,
)


class TestSincosPositionalEmbeddings:
    """Tests for get_3d_sincos_pos_embed."""

    def test_3d_sincos_shape(self):
        """Output shape should be (T, H*W, embed_dim)."""
        embed = get_3d_sincos_pos_embed(
            embed_dim=256,
            spatial_size=(8, 8),
            temporal_size=4,
        )
        mx.eval(embed)
        # T=4, H*W=64, D=256
        assert embed.shape == (4, 64, 256)

    def test_3d_sincos_shape_plan_convention(self):
        """Verify flattened shape matches T*H*W x D."""
        embed = get_3d_sincos_pos_embed(
            embed_dim=256,
            spatial_size=(8, 8),
            temporal_size=4,
        )
        mx.eval(embed)
        flat = embed.reshape(-1, 256)
        assert flat.shape == (256, 256)

    def test_3d_sincos_square_spatial(self):
        """Integer spatial_size should produce a square grid."""
        embed = get_3d_sincos_pos_embed(
            embed_dim=64,
            spatial_size=4,
            temporal_size=2,
        )
        mx.eval(embed)
        assert embed.shape == (2, 16, 64)

    def test_3d_sincos_rectangular(self):
        """Non-square spatial sizes."""
        embed = get_3d_sincos_pos_embed(
            embed_dim=128,
            spatial_size=(4, 8),
            temporal_size=3,
        )
        mx.eval(embed)
        assert embed.shape == (3, 32, 128)

    def test_3d_sincos_bounded(self):
        """Values should be in [-1, 1] since they are sin/cos."""
        embed = get_3d_sincos_pos_embed(
            embed_dim=64,
            spatial_size=4,
            temporal_size=2,
        )
        mx.eval(embed)
        assert float(mx.max(mx.abs(embed))) <= 1.0 + 1e-6

    def test_3d_sincos_interpolation_scale(self):
        """Interpolation scales should change the embedding values."""
        e1 = get_3d_sincos_pos_embed(64, (4, 4), 2, spatial_interpolation_scale=1.0)
        e2 = get_3d_sincos_pos_embed(64, (4, 4), 2, spatial_interpolation_scale=2.0)
        mx.eval(e1, e2)
        assert not mx.allclose(e1, e2, atol=1e-6)

    def test_3d_sincos_embed_dim_not_div4_raises(self):
        with pytest.raises(ValueError, match="divisible by 4"):
            get_3d_sincos_pos_embed(embed_dim=63, spatial_size=4, temporal_size=2)


class TestRotaryPositionalEmbeddings:
    """Tests for get_3d_rotary_pos_embed."""

    def test_3d_rope_shape_linspace(self):
        """Output (cos, sin) should each have shape (T*H*W, embed_dim)."""
        cos, sin = get_3d_rotary_pos_embed(
            embed_dim=64,
            crops_coords=((0, 0), (8, 8)),
            grid_size=(4, 4),
            temporal_size=3,
        )
        mx.eval(cos, sin)
        # T*H*W = 3*4*4 = 48, dim = dim_t + dim_h + dim_w = 16 + 24 + 24 = 64
        assert cos.shape == (48, 64)
        assert sin.shape == (48, 64)

    def test_3d_rope_shape_slice(self):
        cos, sin = get_3d_rotary_pos_embed(
            embed_dim=64,
            crops_coords=((0, 0), (8, 8)),
            grid_size=(4, 4),
            temporal_size=3,
            grid_type="slice",
            max_size=(8, 8),
        )
        mx.eval(cos, sin)
        assert cos.shape == (48, 64)
        assert sin.shape == (48, 64)

    def test_3d_rope_dim_split(self):
        """Verify the dimension split: dim_t = D//4, dim_h = dim_w = D//8*3."""
        embed_dim = 64
        dim_t = embed_dim // 4  # 16
        dim_h = embed_dim // 8 * 3  # 24
        dim_w = embed_dim // 8 * 3  # 24
        assert dim_t + dim_h + dim_w == embed_dim

    def test_3d_rope_cos_sin_bounded(self):
        """cos/sin values should be in [-1, 1]."""
        cos, sin = get_3d_rotary_pos_embed(
            embed_dim=64,
            crops_coords=((0, 0), (16, 16)),
            grid_size=(8, 8),
            temporal_size=4,
        )
        mx.eval(cos, sin)
        assert float(mx.max(mx.abs(cos))) <= 1.0 + 1e-6
        assert float(mx.max(mx.abs(sin))) <= 1.0 + 1e-6

    def test_3d_rope_invalid_grid_type(self):
        with pytest.raises(ValueError, match="Invalid"):
            get_3d_rotary_pos_embed(
                embed_dim=64,
                crops_coords=((0, 0), (8, 8)),
                grid_size=(4, 4),
                temporal_size=3,
                grid_type="bad",
            )


class TestApplyRotaryEmb:
    """Tests for apply_rotary_emb."""

    def test_output_shape(self):
        """Output should match input shape."""
        x = mx.random.normal((2, 4, 48, 64))  # (B, heads, seq, D)
        cos, sin = get_3d_rotary_pos_embed(
            embed_dim=64,
            crops_coords=((0, 0), (8, 8)),
            grid_size=(4, 4),
            temporal_size=3,
        )
        out = apply_rotary_emb(x, (cos, sin))
        mx.eval(out)
        assert out.shape == x.shape

    def test_identity_at_zero_angle(self):
        """When sin=0 and cos=1, output should equal input."""
        x = mx.random.normal((1, 2, 10, 32))
        cos = mx.ones((10, 32))
        sin = mx.zeros((10, 32))
        out = apply_rotary_emb(x, (cos, sin))
        mx.eval(out)
        assert mx.allclose(out, x, atol=1e-6)

    def test_rotation_preserves_norm(self):
        """Rotary embedding should approximately preserve the L2 norm of each vector."""
        x = mx.random.normal((1, 2, 48, 64))
        cos, sin = get_3d_rotary_pos_embed(
            embed_dim=64,
            crops_coords=((0, 0), (8, 8)),
            grid_size=(4, 4),
            temporal_size=3,
        )
        out = apply_rotary_emb(x, (cos, sin))
        mx.eval(out)
        # Norms along last dim
        norm_in = mx.sqrt(mx.sum(x * x, axis=-1))
        norm_out = mx.sqrt(mx.sum(out * out, axis=-1))
        mx.eval(norm_in, norm_out)
        assert mx.allclose(norm_in, norm_out, atol=1e-4)

    def test_different_from_input(self):
        """With non-trivial angles, output should differ from input."""
        x = mx.random.normal((1, 2, 48, 64))
        cos, sin = get_3d_rotary_pos_embed(
            embed_dim=64,
            crops_coords=((0, 0), (8, 8)),
            grid_size=(4, 4),
            temporal_size=3,
        )
        out = apply_rotary_emb(x, (cos, sin))
        mx.eval(out)
        assert not mx.allclose(out, x, atol=1e-6)
