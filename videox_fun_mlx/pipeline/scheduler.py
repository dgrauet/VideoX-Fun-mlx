"""DDIM Scheduler for CogVideoX-Fun MLX port.

Minimal implementation of Denoising Diffusion Implicit Models (DDIM) scheduler,
compatible with CogVideoXDDIMScheduler from diffusers.
"""

import mlx.core as mx


class DDIMScheduler:
    """DDIM scheduler for diffusion model inference.

    Args:
        num_train_timesteps: Number of diffusion steps used during training.
        beta_start: Starting value of beta schedule.
        beta_end: Ending value of beta schedule.
        beta_schedule: Type of beta schedule. Only "scaled_linear" is supported.
        num_inference_steps: Default number of denoising steps for inference.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        num_inference_steps: int = 50,
    ):
        self.num_train_timesteps = num_train_timesteps

        if beta_schedule == "scaled_linear":
            betas = mx.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        elif beta_schedule == "linear":
            betas = mx.linspace(beta_start, beta_end, num_train_timesteps)
        else:
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")

        alphas = 1.0 - betas
        self.alphas_cumprod = mx.cumprod(alphas)

        self._timesteps = None
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Compute the timestep schedule for inference.

        Produces evenly spaced timesteps from num_train_timesteps-1 down to 0.
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        # Timesteps in descending order: [999, 979, 959, ..., 19] for default settings
        timesteps = (
            mx.arange(0, num_inference_steps) * step_ratio
        )
        self._timesteps = timesteps[::-1]

    @property
    def timesteps(self) -> mx.array:
        """Return the current timestep schedule."""
        return self._timesteps

    def step(
        self,
        model_output: mx.array,
        timestep: mx.array,
        sample: mx.array,
    ) -> mx.array:
        """Perform one DDIM denoising step.

        Args:
            model_output: Predicted noise from the model (epsilon prediction).
            timestep: Current timestep (scalar).
            sample: Current noisy sample.

        Returns:
            Denoised sample after one DDIM step.
        """
        # Current timestep index
        t = int(timestep.item()) if isinstance(timestep, mx.array) else int(timestep)
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        prev_t = t - step_ratio

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else mx.array(1.0)

        # Predict x_0
        sqrt_alpha_prod_t = mx.sqrt(alpha_prod_t)
        sqrt_one_minus_alpha_prod_t = mx.sqrt(1.0 - alpha_prod_t)
        pred_x0 = (sample - sqrt_one_minus_alpha_prod_t * model_output) / sqrt_alpha_prod_t

        # Compute predicted sample (deterministic DDIM, eta=0)
        sqrt_alpha_prod_t_prev = mx.sqrt(alpha_prod_t_prev)
        sqrt_one_minus_alpha_prod_t_prev = mx.sqrt(1.0 - alpha_prod_t_prev)
        pred_sample = sqrt_alpha_prod_t_prev * pred_x0 + sqrt_one_minus_alpha_prod_t_prev * model_output

        return pred_sample

    def add_noise(
        self,
        original: mx.array,
        noise: mx.array,
        timestep: mx.array,
    ) -> mx.array:
        """Add noise to original samples at the given timestep level.

        Args:
            original: Clean samples.
            noise: Noise to add (same shape as original).
            timestep: Timestep controlling the noise level (scalar).

        Returns:
            Noisy samples.
        """
        t = int(timestep.item()) if isinstance(timestep, mx.array) else int(timestep)
        alpha_prod_t = self.alphas_cumprod[t]

        sqrt_alpha_prod = mx.sqrt(alpha_prod_t)
        sqrt_one_minus_alpha_prod = mx.sqrt(1.0 - alpha_prod_t)

        return sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise
