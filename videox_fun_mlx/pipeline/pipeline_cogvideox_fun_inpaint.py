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
        text_encoder=None,
        tokenizer=None,
    ):
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    def encode_prompt(self, prompt: str, max_length: int = 226) -> mx.array:
        """Encode a text prompt to embeddings using T5.

        Args:
            prompt: Text string.
            max_length: Max token length.

        Returns:
            (1, max_length, d_model) text embeddings.
        """
        if self.tokenizer is None or self.text_encoder is None:
            raise RuntimeError("text_encoder and tokenizer required for prompt encoding")
        input_ids = self.tokenizer(prompt, max_length=max_length)
        return self.text_encoder(input_ids)

    def __call__(
        self,
        video: mx.array,
        mask: mx.array,
        prompt: Optional[str] = None,
        prompt_embeds: Optional[mx.array] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        seed: Optional[int] = None,
    ) -> mx.array:
        """Run video inpainting.

        Args:
            video: (B, D, H, W, C) input video in channels-last.
            mask: (B, D, H, W, 1) binary mask (1 = inpaint region).
            prompt: Text prompt (requires text_encoder + tokenizer).
            prompt_embeds: (B, text_len, text_dim) pre-computed text embeddings.
                Either prompt or prompt_embeds must be provided.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale (unused for now).
            seed: Random seed.

        Returns:
            (B, D, H, W, C) inpainted video.
        """
        if prompt_embeds is None:
            if prompt is None:
                raise ValueError("Either prompt or prompt_embeds must be provided")
            prompt_embeds = self.encode_prompt(prompt)

        if seed is not None:
            mx.random.seed(seed)

        B = video.shape[0]

        # 1. Encode video to latent space
        posterior = self.vae.encode(video)
        latents = posterior.mode() * self.vae.scaling_factor

        # Determine latent shape for noise generation
        latent_cf = latents.transpose(0, 1, 4, 2, 3)  # NDHWC -> NFCHW
        B_lat, F_lat, C_lat, H_lat, W_lat = latent_cf.shape

        # 2. Prepare inpaint conditioning
        is_full_mask = mx.mean(mask).item() > 0.99
        if is_full_mask:
            # Full mask = generate from scratch: use zeros for conditioning
            mask_latent_1ch = mx.zeros((B_lat, F_lat, 1, H_lat, W_lat))
            masked_video_latents_cf = mx.zeros((B_lat, F_lat, C_lat, H_lat, W_lat))
        else:
            # Partial mask: encode masked video, resize mask
            masked_video = video * (1 - mask)
            masked_posterior = self.vae.encode(masked_video)
            masked_video_latents = masked_posterior.mode() * self.vae.scaling_factor
            masked_video_latents_cf = masked_video_latents.transpose(0, 1, 4, 2, 3)
            mask_latent = _resize_mask_to_latent(mask, latents.shape)
            mask_latent_1ch = mask_latent.transpose(0, 1, 4, 2, 3)

        inpaint_cf = mx.concatenate([mask_latent_1ch, masked_video_latents_cf], axis=2)

        # 3. Setup scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # 4. Start from noise
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
        """Load pipeline from a pretrained model directory.

        Loads VAE, transformer, T5 text encoder, and tokenizer.
        """
        import os
        from videox_fun_mlx.models.t5_encoder import T5Encoder
        from videox_fun_mlx.models.tokenizer import T5Tokenizer

        vae_path = os.path.join(model_path, "vae")
        tf_path = os.path.join(model_path, "transformer")

        # VAE and transformer can be in subdirs or flat (mlx-forge output)
        if os.path.isdir(vae_path):
            vae = AutoencoderKLCogVideoX.from_pretrained(vae_path)
        else:
            vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        if os.path.isdir(tf_path):
            transformer = CogVideoXTransformer3DModel.from_pretrained(tf_path)
        else:
            transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer")

        # T5 encoder (optional — might not be present)
        text_encoder = None
        tokenizer = None
        t5_weights = os.path.join(model_path, "text_encoder.safetensors")
        spiece_file = os.path.join(model_path, "tokenizer_spiece.model")
        if os.path.exists(t5_weights):
            print("Loading T5 text encoder...")
            text_encoder = T5Encoder.from_pretrained(model_path)
        if os.path.exists(spiece_file):
            tokenizer = T5Tokenizer(model_path)

        # Load scheduler config if available
        scheduler_config = {}
        sched_config_file = os.path.join(model_path, "scheduler_scheduler_config.json")
        if os.path.exists(sched_config_file):
            import json
            with open(sched_config_file) as f:
                scheduler_config = json.load(f)
            # Remove diffusers-internal keys
            scheduler_config.pop("_class_name", None)
            scheduler_config.pop("_diffusers_version", None)
            scheduler_config.pop("trained_betas", None)
        scheduler_config.update(kwargs)
        scheduler = DDIMScheduler(**scheduler_config)

        return cls(
            vae=vae, transformer=transformer, scheduler=scheduler,
            text_encoder=text_encoder, tokenizer=tokenizer,
        )
