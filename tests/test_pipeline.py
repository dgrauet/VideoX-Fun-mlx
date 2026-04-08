"""Tests for the DDIM scheduler."""

import mlx.core as mx
from videox_fun_mlx.pipeline.scheduler import DDIMScheduler


class TestDDIMScheduler:
    def test_timesteps_length(self):
        """Scheduler produces correct number of timesteps."""
        sched = DDIMScheduler(num_train_timesteps=1000, num_inference_steps=50)
        assert len(sched.timesteps) == 50

    def test_timesteps_descending(self):
        """Timesteps are in descending order."""
        sched = DDIMScheduler(num_train_timesteps=1000, num_inference_steps=50)
        ts = sched.timesteps.tolist()
        assert ts == sorted(ts, reverse=True)

    def test_timesteps_range(self):
        """First timestep should be near num_train_timesteps-1, last near 0."""
        sched = DDIMScheduler(num_train_timesteps=1000, num_inference_steps=50)
        ts = sched.timesteps.tolist()
        assert ts[0] == 980  # (50-1)*20 = 980
        assert ts[-1] == 0

    def test_set_timesteps(self):
        """set_timesteps updates the schedule."""
        sched = DDIMScheduler(num_train_timesteps=1000, num_inference_steps=50)
        assert len(sched.timesteps) == 50
        sched.set_timesteps(10)
        assert len(sched.timesteps) == 10

    def test_step_shape(self):
        """step() returns output with same shape as input."""
        sched = DDIMScheduler(num_train_timesteps=1000, num_inference_steps=50)
        sample = mx.random.normal((1, 4, 4, 4, 16))
        noise_pred = mx.random.normal((1, 4, 4, 4, 16))
        out = sched.step(noise_pred, sched.timesteps[0], sample)
        mx.eval(out)
        assert out.shape == sample.shape

    def test_step_reduces_noise(self):
        """After a step from a high timestep, output should differ from input."""
        sched = DDIMScheduler(num_train_timesteps=1000, num_inference_steps=50)
        sample = mx.random.normal((1, 4, 4, 4, 16))
        noise_pred = mx.random.normal((1, 4, 4, 4, 16))
        out = sched.step(noise_pred, sched.timesteps[0], sample)
        mx.eval(out)
        # Output should not be identical to input
        diff = mx.abs(out - sample).max().item()
        assert diff > 0.0

    def test_add_noise_shape(self):
        """add_noise returns correct shape."""
        sched = DDIMScheduler(num_train_timesteps=1000, num_inference_steps=50)
        original = mx.random.normal((1, 4, 4, 4, 16))
        noise = mx.random.normal((1, 4, 4, 4, 16))
        noisy = sched.add_noise(original, noise, mx.array(500))
        mx.eval(noisy)
        assert noisy.shape == original.shape

    def test_add_noise_at_zero(self):
        """At t=0, add_noise should return nearly the original (alpha ~1)."""
        sched = DDIMScheduler(num_train_timesteps=1000, num_inference_steps=50)
        original = mx.random.normal((1, 4, 4, 4, 16))
        noise = mx.random.normal((1, 4, 4, 4, 16))
        noisy = sched.add_noise(original, noise, mx.array(0))
        mx.eval(noisy)
        # At t=0, alpha_cumprod[0] is very close to 1, so noisy ~ original
        diff = mx.abs(noisy - original).max().item()
        assert diff < 0.15

    def test_add_noise_at_high_t(self):
        """At high t, add_noise should produce mostly noise."""
        sched = DDIMScheduler(num_train_timesteps=1000, num_inference_steps=50)
        original = mx.zeros((1, 4, 4, 4, 16))
        noise = mx.ones((1, 4, 4, 4, 16))
        noisy = sched.add_noise(original, noise, mx.array(999))
        mx.eval(noisy)
        # At t=999, alpha_cumprod is very small, so noisy ~ noise
        mean_val = mx.mean(noisy).item()
        assert mean_val > 0.5  # mostly noise

    def test_alphas_cumprod_shape(self):
        """alphas_cumprod has correct length."""
        sched = DDIMScheduler(num_train_timesteps=1000)
        assert sched.alphas_cumprod.shape == (1000,)

    def test_alphas_cumprod_monotonic(self):
        """alphas_cumprod should be monotonically decreasing."""
        sched = DDIMScheduler(num_train_timesteps=1000)
        acp = sched.alphas_cumprod
        mx.eval(acp)
        diffs = acp[1:] - acp[:-1]
        mx.eval(diffs)
        assert mx.all(diffs < 0).item()

    def test_linear_beta_schedule(self):
        """Linear beta schedule should also work."""
        sched = DDIMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            num_inference_steps=10,
        )
        assert len(sched.timesteps) == 10
        assert sched.alphas_cumprod.shape == (100,)

    def test_full_denoising_loop(self):
        """Run a full denoising loop to verify no errors."""
        sched = DDIMScheduler(num_train_timesteps=1000, num_inference_steps=5)
        sample = mx.random.normal((1, 2, 4, 4, 8))

        for t in sched.timesteps:
            noise_pred = mx.random.normal(sample.shape)
            sample = sched.step(noise_pred, t, sample)
            mx.eval(sample)

        assert sample.shape == (1, 2, 4, 4, 8)
