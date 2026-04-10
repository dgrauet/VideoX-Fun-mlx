#!/usr/bin/env python3
"""Quick inference demo -- generates a short video from noise + text prompt.

Usage:
    python scripts/quick_infer.py --prompt "a sunset over the ocean"
    python scripts/quick_infer.py --prompt "a cat" --steps 10 --output cat.gif
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="CogVideoX-Fun MLX quick inference")
    parser.add_argument("--prompt", type=str, default="a beautiful sunset over the ocean")
    parser.add_argument("--model-path", type=str,
                        default="/Users/dgrauet/Work/mlx-forge/models/cogvideox-fun-v1.5-5b-inp-mlx")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output.gif")
    args = parser.parse_args()

    from videox_fun_mlx.models.cogvideox_vae import AutoencoderKLCogVideoX
    from videox_fun_mlx.models.cogvideox_transformer3d import CogVideoXTransformer3DModel
    from videox_fun_mlx.models.t5_encoder import T5Encoder
    from videox_fun_mlx.models.tokenizer import T5Tokenizer
    from videox_fun_mlx.pipeline.pipeline_cogvideox_fun_inpaint import CogVideoXFunInpaintPipeline
    from videox_fun_mlx.pipeline.scheduler import DDIMScheduler

    total_t0 = time.monotonic()

    # Load models
    print(f"Loading VAE from {args.model_path}...")
    t0 = time.monotonic()
    vae = AutoencoderKLCogVideoX.from_pretrained(args.model_path)
    print(f"  done ({time.monotonic() - t0:.1f}s)")

    print(f"Loading transformer from {args.model_path}...")
    t0 = time.monotonic()
    transformer = CogVideoXTransformer3DModel.from_pretrained(args.model_path)
    print(f"  done ({time.monotonic() - t0:.1f}s)")

    print(f"Loading T5 text encoder from {args.model_path}...")
    t0 = time.monotonic()
    t5 = T5Encoder.from_pretrained(args.model_path)
    tok = T5Tokenizer(args.model_path)
    print(f"  done ({time.monotonic() - t0:.1f}s)")

    scheduler = DDIMScheduler(num_inference_steps=args.steps)
    pipe = CogVideoXFunInpaintPipeline(
        vae=vae, transformer=transformer, scheduler=scheduler,
        text_encoder=t5, tokenizer=tok,
    )

    H, W, F = args.height, args.width, args.frames
    print(f"\nGenerating {F} frames at {H}x{W}, {args.steps} steps, guidance={args.guidance_scale}")
    print(f'Prompt: "{args.prompt}"')

    video = mx.zeros((1, F, H, W, 3))
    mask = mx.ones((1, F, H, W, 1))

    t0 = time.monotonic()
    output = pipe(
        video=video,
        mask=mask,
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    mx.eval(output)
    gen_time = time.monotonic() - t0
    print(f"Generation: {gen_time:.1f}s ({gen_time/args.steps:.1f}s/step)")

    # Pipeline output is already in [0, 1]
    output_np = np.array(output[0].astype(mx.float32))
    print(f"Shape: {output_np.shape}, range: [{output_np.min():.3f}, {output_np.max():.3f}]")
    output_np = (output_np * 255).clip(0, 255).astype(np.uint8)

    from PIL import Image
    frames_pil = [Image.fromarray(output_np[i]) for i in range(output_np.shape[0])]
    frames_pil[0].save(
        args.output,
        save_all=True,
        append_images=frames_pil[1:],
        duration=200,
        loop=0,
    )

    print(f"\nSaved {len(frames_pil)} frames to {args.output}")
    print(f"Total: {time.monotonic() - total_t0:.1f}s")


if __name__ == "__main__":
    main()
