"""bf16 dtype-flow contracts (VOID dogfood 2026-07-20).

The reference runs weight_dtype = torch.bfloat16 end-to-end (videox_fun
inference scripts) with two deliberate fp32 islands, each followed by a cast
back:
  - diffusers apply_rotary_emb: (x.float()*cos + x_rot.float()*sin).to(x.dtype)
  - cogvideox_transformer3d.py:597: t_emb = t_emb.to(hidden_states.dtype)
The MLX port dropped both casts and let fp32 alphas/cos contaminate the whole
forward (observed: 182 s of float32 gemms + 71k bf16->fp32 weight upcasts on
a 30-step run, smeltr session void-dogfood-bf16).
"""

import mlx.core as mx
import numpy as np

from videox_fun_mlx.models.embeddings import apply_rotary_emb
from videox_fun_mlx.pipeline.scheduler import DDIMScheduler


class TestRotaryEmbDtype:
    def test_bf16_input_stays_bf16(self):
        mx.random.seed(0)
        x = mx.random.normal((1, 2, 6, 8)).astype(mx.bfloat16)
        cos = mx.random.normal((6, 8))  # fp32 tables, like the reference
        sin = mx.random.normal((6, 8))
        out = apply_rotary_emb(x, (cos, sin))
        assert out.dtype == mx.bfloat16

    def test_matches_fp32_island_semantics(self):
        """Mirror of diffusers: compute in fp32, cast the RESULT to x.dtype."""
        mx.random.seed(1)
        x = mx.random.normal((1, 2, 6, 8)).astype(mx.bfloat16)
        cos = mx.random.normal((6, 8))
        sin = mx.random.normal((6, 8))

        out = apply_rotary_emb(x, (cos, sin))

        xf = x.astype(mx.float32)
        paired = xf.reshape(1, 2, 6, 4, 2)
        rot = mx.stack([-paired[..., 1], paired[..., 0]], axis=-1).reshape(x.shape)
        want = (xf * cos + rot * sin).astype(mx.bfloat16)
        assert np.array_equal(np.array(out.astype(mx.float32)), np.array(want.astype(mx.float32)))


class TestSchedulerDtype:
    def test_add_noise_preserves_bf16(self):
        sched = DDIMScheduler()
        sched.set_timesteps(10)
        mx.random.seed(2)
        x = mx.random.normal((1, 2, 4, 4, 4)).astype(mx.bfloat16)
        noise = mx.random.normal((1, 2, 4, 4, 4)).astype(mx.bfloat16)
        noisy = sched.add_noise(x, noise, sched.timesteps[0])
        assert noisy.dtype == mx.bfloat16

    def test_step_promotes_to_fp32_like_torch(self):
        """ISO-UPSTREAM: diffusers DDIM step computes with fp32 alphas and
        torch promotion yields fp32; the reference pipeline then casts
        latents back to the model dtype after every step
        (pipeline_cogvideox_fun_inpaint.py:1170). Do NOT "fix" the
        scheduler to stay bf16 — the cast belongs to the caller."""
        sched = DDIMScheduler()
        sched.set_timesteps(10)
        mx.random.seed(3)
        sample = mx.random.normal((1, 2, 4, 4, 4)).astype(mx.bfloat16)
        model_out = mx.random.normal((1, 2, 4, 4, 4)).astype(mx.bfloat16)
        prev = sched.step(model_out, sched.timesteps[0], sample)
        prev_sample = prev[0] if isinstance(prev, tuple) else prev
        assert prev_sample.dtype == mx.float32


class TestTransformerDtype:
    def test_tiny_forward_is_bf16_end_to_end(self):
        from videox_fun_mlx.models.cogvideox_transformer3d import (
            CogVideoXTransformer3DModel,
        )

        model = CogVideoXTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=4,
            out_channels=4,
            time_embed_dim=16,
            text_embed_dim=12,
            num_layers=1,
            sample_width=8,
            sample_height=8,
            sample_frames=8,
            patch_size=2,
            max_text_seq_length=3,
            use_rotary_positional_embeddings=False,
        )
        model.set_dtype(mx.bfloat16)

        # sample_frames=8, compression 4 -> pos-embed grid for 2 latent frames.
        hidden = mx.random.normal((1, 2, 4, 8, 8)).astype(mx.bfloat16)
        text = mx.random.normal((1, 3, 12)).astype(mx.bfloat16)
        out = model(
            hidden_states=hidden,
            encoder_hidden_states=text,
            timestep=mx.array([500]),
        )
        assert out.dtype == mx.bfloat16, "fp32 leaked into the forward (t_emb island cast missing?)"
