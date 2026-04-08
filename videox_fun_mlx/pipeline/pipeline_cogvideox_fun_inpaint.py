"""CogVideoX-Fun Inpaint Pipeline for MLX.

Orchestrates VAE encoding, transformer denoising, and VAE decoding
for video inpainting. Accepts pre-computed text embeddings (no T5 dependency).
"""

from typing import Optional, Tuple

import mlx.core as mx
import numpy as np

from videox_fun_mlx.models.cogvideox_vae import AutoencoderKLCogVideoX
from videox_fun_mlx.models.cogvideox_transformer3d import CogVideoXTransformer3DModel
from videox_fun_mlx.models.embeddings import get_3d_rotary_pos_embed
from videox_fun_mlx.pipeline.scheduler import DDIMScheduler


def _resize_mask_to_latent(mask: mx.array, latent_shape: tuple) -> mx.array:
    """Resize a binary mask to match latent spatial dimensions via nearest neighbor.

    Args:
        mask: (B, D, H, W, 1) binary mask in pixel space.
        latent_shape: Target shape (B, D_lat, H_lat, W_lat, C_lat).

    Returns:
        (B, D_lat, H_lat, W_lat, 1) resized mask.
    """
    B, D, H, W, _ = mask.shape
    _, D_t, H_t, W_t, _ = latent_shape

    mask_np = np.array(mask)
    d_idx = np.round(np.linspace(0, D - 1, D_t)).astype(int)
    h_idx = np.round(np.linspace(0, H - 1, H_t)).astype(int)
    w_idx = np.round(np.linspace(0, W - 1, W_t)).astype(int)
    resized = mask_np[:, d_idx][:, :, h_idx][:, :, :, w_idx]
    return mx.array(resized)


class CogVideoXFunInpaintPipeline:
    """Video inpainting pipeline for CogVideoX-Fun on MLX.

    Args:
        vae: AutoencoderKLCogVideoX model.
        transformer: CogVideoXTransformer3DModel model.
        scheduler: DDIMScheduler instance.
    """

    def __init__(
        self,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: DDIMScheduler,
    ):
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler

    def __call__(
        self,
        prompt_embeds: mx.array,
        video: mx.array,
        mask: mx.array,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        seed: Optional[int] = None,
    ) -> mx.array:
        """Run video inpainting.

        Args:
            prompt_embeds: (B, text_len, text_dim) pre-computed text embeddings.
            video: (B, D, H, W, C) input video in channels-last.
            mask: (B, D, H, W, 1) binary mask (1 = inpaint region).
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale (unused for now).
            seed: Random seed.

        Returns:
            (B, D, H, W, C) inpainted video.
        """
        if seed is not None:
            mx.random.seed(seed)

        B = video.shape[0]

        # 1. Encode video to latents (VAE accepts NDHWC)
        posterior = self.vae.encode(video)
        latents = posterior.sample()
        latents = latents * self.vae.scaling_factor

        # 2. Prepare masked video latents
        masked_video = video * (1 - mask)
        masked_posterior = self.vae.encode(masked_video)
        masked_video_latents = masked_posterior.mode() * self.vae.scaling_factor

        # 3. Resize mask to latent space
        mask_latents = _resize_mask_to_latent(mask, latents.shape)

        # 4. Concatenate mask + masked_video_latents for inpainting conditioning
        inpaint_latents = mx.concatenate([mask_latents, masked_video_latents], axis=-1)

        # Convert to channels-first for transformer: (B, D, H, W, C) -> (B, F, C, H, W)
        latent_cf = latents.transpose(0, 1, 4, 2, 3)
        inpaint_cf = inpaint_latents.transpose(0, 1, 4, 2, 3)

        # 5. Setup scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # 6. Add noise to latents
        noise = mx.random.normal(latent_cf.shape)
        noisy_latents = self.scheduler.add_noise(latent_cf, noise, self.scheduler.timesteps[0])

        # 7. Compute RoPE if transformer uses it
        image_rotary_emb = None
        if self.transformer._config.get("use_rotary_positional_embeddings"):
            _, F, C, H, W = latent_cf.shape
            p = self.transformer._config["patch_size"]
            p_t = self.transformer._config.get("patch_size_t")
            grid_h = H // p
            grid_w = W // p
            grid_t = (F + p_t - 1) // p_t if p_t is not None else F

            head_dim = self.transformer.transformer_blocks[0].attn1.dim_head
            image_rotary_emb = get_3d_rotary_pos_embed(
                embed_dim=head_dim,
                crops_coords=((0, 0), (grid_h, grid_w)),
                grid_size=(grid_h, grid_w),
                temporal_size=grid_t,
            )

        # 8. Denoising loop
        current = noisy_latents
        for i, t in enumerate(self.scheduler.timesteps):
            t_input = mx.array([float(t)])

            noise_pred = self.transformer(
                hidden_states=current,
                encoder_hidden_states=prompt_embeds,
                timestep=t_input,
                inpaint_latents=inpaint_cf,
                image_rotary_emb=image_rotary_emb,
            )

            current = self.scheduler.step(noise_pred, t, current)

        # 9. Decode latents
        decoded_latents = current.transpose(0, 1, 3, 4, 2)  # NFCHW -> NDHWC
        decoded_latents = decoded_latents / self.vae.scaling_factor
        output = self.vae.decode(decoded_latents)

        return output

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load pipeline from a pretrained model directory."""
        import os

        vae = AutoencoderKLCogVideoX.from_pretrained(os.path.join(model_path, "vae"))
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            os.path.join(model_path, "transformer")
        )
        scheduler = DDIMScheduler(**kwargs)

        return cls(vae=vae, transformer=transformer, scheduler=scheduler)
