# CogVideoX-Fun MLX

A port of [CogVideoX-Fun-V1.5-5b-InP](https://github.com/aigc-apps/VideoX-Fun) to Apple [MLX](https://github.com/ml-explore/mlx) for video generation and inpainting on Apple Silicon.

## Features

- **Text-to-video generation** -- generate short videos from a text prompt
- **Video inpainting** -- fill in masked regions of an existing video
- Runs entirely on-device using Apple Silicon GPU via MLX
- Quantized model variants to fit different memory budgets

## Models

Available on HuggingFace:

| Model | Format | VRAM |
|---|---|---|
| [dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx](https://huggingface.co/dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx) | bf16 | ~20 GB |
| [dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q8](https://huggingface.co/dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q8) | int8 | ~16 GB |
| [dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q4](https://huggingface.co/dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q4) | int4 | ~14 GB |

## Requirements

- Apple Silicon Mac (M1 or later)
- Python 3.10+
- MLX 0.27+

## Quick Start

1. **Install dependencies:**

```bash
pip install mlx sentencepiece pillow numpy
pip install git+https://github.com/dgrauet/mlx-arsenal.git
```

2. **Download a model** (using the HuggingFace CLI or any method you prefer):

```bash
pip install huggingface_hub
huggingface-cli download dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q8 --local-dir models/cogvideox-fun-q8
```

3. **Generate a video:**

```bash
python scripts/quick_infer.py \
    --model-path models/cogvideox-fun-q8 \
    --prompt "a beautiful sunset over the ocean" \
    --output sunset.gif
```

The output is saved as an animated GIF.

## Script Options

```
python scripts/quick_infer.py [OPTIONS]

--prompt          Text description of the video to generate
--model-path      Path to the MLX model directory
--steps           Number of diffusion steps (default: 50)
--guidance-scale  Classifier-free guidance scale (default: 6.0)
--height          Output height in pixels (default: 384)
--width           Output width in pixels (default: 672)
--frames          Number of frames to generate (default: 5)
--seed            Random seed (default: 42)
--output          Output file path (default: output.gif)
```

## Project Structure

```
videox_fun_mlx/
  models/
    cogvideox_transformer3d.py   # DiT backbone
    cogvideox_vae.py             # 3D VAE encoder/decoder
    t5_encoder.py                # T5 text encoder
    tokenizer.py                 # T5 tokenizer
  pipeline/
    pipeline_cogvideox_fun_inpaint.py  # Inpainting pipeline
    scheduler.py                       # DDIM scheduler
  utils.py
scripts/
  quick_infer.py                 # CLI inference script
```

## Related Projects

- [mlx-forge](https://github.com/dgrauet/mlx-forge) -- tools for porting models to MLX
- [mlx-arsenal](https://github.com/dgrauet/mlx-arsenal) -- custom MLX operations
- [void-model-mlx](https://github.com/dgrauet/void-model-mlx) -- other MLX model ports

## Credits

Based on [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) by [alibaba-pai](https://github.com/aigc-apps).

## License

Apache 2.0 -- see [LICENSE](LICENSE).
