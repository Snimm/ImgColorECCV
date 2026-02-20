#!/bin/bash
export PYTHONPATH=""
# Isolate from system python packages
export PYTHONNOUSERSITE=1

# Use the virtual environment python created by uv
PYTHON_EXEC="$(pwd)/.uv_venv/bin/python"

# Launch latent caching
echo "Launching latent caching with accelerate"
$PYTHON_EXEC -m accelerate.commands.accelerate_cli launch --config_file accelerate_config.yaml \
    musubi-tuner/flux_2_cache_latents.py \
    --dataset_config dataset_config.toml \
    --vae models/flux_klein_4b/ae.safetensors \
    --model_version klein-4b \
    --batch_size 4 \
    --skip_existing

# Launch text encoder caching
echo "Launching text encoder caching with accelerate"
$PYTHON_EXEC -m accelerate.commands.accelerate_cli launch --config_file accelerate_config.yaml \
    musubi-tuner/flux_2_cache_text_encoder_outputs.py \
    --dataset_config dataset_config.toml \
    --text_encoder models/flux_klein_4b/text_encoder/model-00001-of-00002.safetensors \
    --model_version klein-4b \
    --batch_size 4 \
    --skip_existing

# Launch Training automatically
echo "Caching complete. Starting Training..."
./start_training.sh
