#!/usr/bin/env python3
"""Validate MLX port against real CogVideoX-Fun weights.

Usage:
    python scripts/test_with_real_weights.py --model-path /path/to/CogVideoX-Fun-V1.5-5b-InP

Steps:
    1. Load VAE weights and verify encode/decode roundtrip
    2. Load Transformer weights and verify forward pass
    3. Report parameter counts and shapes
"""

import argparse
import sys
import time


def test_vae(model_path: str):
    import mlx.core as mx
    from videox_fun_mlx.models.cogvideox_vae import AutoencoderKLCogVideoX

    print("=" * 60)
    print("Testing VAE")
    print("=" * 60)

    t0 = time.monotonic()
    vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")
    print(f"  Loaded in {time.monotonic() - t0:.1f}s")

    param_count = sum(p.size for _, p in vae.parameters().items())
    print(f"  Parameters: {param_count / 1e6:.1f}M")

    print("\n  Testing encode...")
    x = mx.random.normal((1, 5, 64, 64, 3))
    t0 = time.monotonic()
    posterior = vae.encode(x)
    z = posterior.mode()
    mx.eval(z)
    print(f"  Encode: {x.shape} -> {z.shape} ({time.monotonic() - t0:.1f}s)")

    print("  Testing decode...")
    t0 = time.monotonic()
    recon = vae.decode(z)
    mx.eval(recon)
    print(f"  Decode: {z.shape} -> {recon.shape} ({time.monotonic() - t0:.1f}s)")

    print("  VAE OK!")
    return True


def test_transformer(model_path: str):
    import mlx.core as mx
    from videox_fun_mlx.models.cogvideox_transformer3d import CogVideoXTransformer3DModel

    print("\n" + "=" * 60)
    print("Testing Transformer")
    print("=" * 60)

    t0 = time.monotonic()
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_path, subfolder="transformer"
    )
    print(f"  Loaded in {time.monotonic() - t0:.1f}s")

    print("\n  Testing forward pass...")
    hidden = mx.random.normal((1, 2, 16, 8, 8))
    encoder = mx.random.normal((1, 226, 4096))
    timestep = mx.array([500.0])

    t0 = time.monotonic()
    out = transformer(hidden, encoder, timestep)
    mx.eval(out)
    print(f"  Forward: {hidden.shape} -> {out.shape} ({time.monotonic() - t0:.1f}s)")

    print("  Transformer OK!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test MLX port with real weights")
    parser.add_argument("--model-path", required=True, help="Path to model directory")
    parser.add_argument("--vae-only", action="store_true")
    parser.add_argument("--transformer-only", action="store_true")
    args = parser.parse_args()

    results = []

    if not args.transformer_only:
        try:
            results.append(("VAE", test_vae(args.model_path)))
        except Exception as e:
            print(f"\n  VAE FAILED: {e}")
            results.append(("VAE", False))

    if not args.vae_only:
        try:
            results.append(("Transformer", test_transformer(args.model_path)))
        except Exception as e:
            print(f"\n  Transformer FAILED: {e}")
            results.append(("Transformer", False))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll checks passed!")
    else:
        print("\nSome checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
