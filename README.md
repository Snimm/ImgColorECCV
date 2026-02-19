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

---

# Fine-tuning Flux Klein 4B for Image Colorization

This repository contains the complete pipeline for fine-tuning **Flux Klein 4B** to perform image colorization (Grayscale -> Color). It uses `musubi-tuner` with custom modifications for on-the-fly grayscale conversion.

## 1. Environment Setup

We use `uv` for fast environment management.

```bash
# Install dependencies
uv python install -r requirements.txt
```

Ensure you have the `musubi-tuner` directory (included in this repo) as it contains necessary source code modifications.

## 2. Data Preparation

### A. Download ImageNet
The training assumes you have the ImageNet dataset (or a subset like `imagenet-mini`).
You can use the provided script (requires Kaggle API token):

```bash
./download_data.sh
```

### B. Caption Images
We use BLIP-large/BLIP-2 to generate captions for the training images. This is crucial for text-conditioned models like Flux.

```bash
uv run python caption_images.py --data_dir /path/to/imagenet/train
```

### C. Create Dataset JSONL
Convert your image directory and captions into a `.jsonl` file compatible with `musubi-tuner`.

```bash
uv run python preprocess_dataset.py \
    --data_dir /path/to/imagenet/train \
    --output_file dataset.jsonl
```

The script automatically handles:
- Relative pathing
- JSONL formatting
- Train/Val splitting (optional)

## 3. Configuration

The training is controlled by `train_config.toml` and `dataset_config.toml`.

- **`dataset_config.toml`**: Defines resolution (`512x512`), bucket settings, and the path to `dataset.jsonl`. Crucially, it enables `auto_grayscale_control = true` (custom feature) to generate control images on the fly.
- **`train_config.toml`**: Sets hyperparameters like learning rate (`1e-4`), batch size, LoRA rank (`16`), and checkpointing frequency.

## 4. Running the Training Pipeline

We provide a master script `run_pipeline.sh` that automates the entire process:
1.  **Caching**: Pre-computes latents and text encodings to multi-GPU caching for speed (approx. 4x faster training).
2.  **Training**: Launches the distributed training on all available GPUs using `accelerate`.

```bash
# Make executable
chmod +x run_pipeline.sh

# Start the pipeline
./run_pipeline.sh
```

**Manual Start:**
If you want to run just the training step:
```bash
./start_training.sh
```

## 5. Monitoring & Evaluation

### Real-time Monitoring
We include a background monitoring script that tracks training progress and runs FID evaluation.

```bash
uv run python monitor_and_eval.py
```
This script:
- Watches for new checkpoints.
- Generates validation images.
- Calculates FID scores.
- Implements Early Stopping logic.

### Inference / Testing
To test a specific checkpoint on a folder of images:

```bash
uv run python test_imagenet5k.py \
    --checkpoint /path/to/checkpoint.safetensors \
    --image_dir /path/to/test/images \
    --output_dir ./results
```

This script:
1.  Reads input images.
2.  Generates BLIP2 captions (conditional text).
3.  Converts images to Grayscale (Control Input).
4.  Runs Flux Klein Io produce the colorized output.
5.  Saves the result as `[filename]_colorized.jpg`.

## 6. Project Structure

- `musubi-tuner/`: Core training library (modified).
- `train_config.toml`: Training hyperparameters.
- `accelerate_config.yaml`: Multi-GPU setup.
- `test_imagenet5k.py`: main inference script.
- `preprocess_dataset.py`: Data preparation tool.
