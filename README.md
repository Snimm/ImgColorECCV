# FLUX.2-klein-4B Image Generation (UV Managed)

Fast, high-quality image generation using FLUX.2-klein-4B on GPU, managed with `uv`.

## Quick Start

```bash
# Text-to-Image
uv run python flux_klein.py --prompt "A serene mountain landscape at sunset"

# Image-to-Image (Image Conditioning)
uv run python flux_klein.py --image "input.png" --prompt "Change the scene to daytime"
```

## Usage

```bash
uv run python flux_klein.py \
    --prompt "Your creative prompt here" \
    --image "optional_input.png" \
    --height 1024 \
    --width 1024 \
    --steps 4 \
    --guidance 1.0 \
    --seed 42 \
    --output "my_image.png"
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--prompt` | (required) | Text description of image to generate |
| `--output` | auto | Output file path |
| `--height` | 1024 | Image height in pixels |
| `--width` | 1024 | Image width in pixels |
| `--steps` | 4 | Number of inference steps (4-8 recommended) |
| `--guidance` | 1.0 | Guidance scale (1.0 recommended for klein) |
| `--seed` | 42 | Random seed for reproducibility |

## Environment Management with UV

This project uses `uv` for lightning-fast and reproducible environment management.

### Installation on a new system:
1. Install `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Clone/Copy the project files (`pyproject.toml`, `uv.lock`, `flux_klein.py`)
3. Run any command with `uv run`, and it will automatically set up the environment.

### Adding dependencies:
```bash
uv add <package_name>
```

## Performance

- **512x512**: ~0.5 seconds
- **1024x1024**: ~1-2 seconds  
- **GPU**: NVIDIA RTX A6000
- **VRAM**: ~13GB

---

**Ready to use!** Generate your first image now. 🎨
