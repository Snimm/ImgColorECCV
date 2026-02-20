#!/bin/bash
export PYTHONPATH=""
# Isolate from system python packages
export PYTHONNOUSERSITE=1

# Use the virtual environment python created by uv
PYTHON_EXEC="$(pwd)/.uv_venv/bin/python"

# Launch text encoder caching
echo "Launching text encoder caching with $PYTHON_EXEC"
$PYTHON_EXEC musubi-tuner/flux_2_cache_text_encoder_outputs.py \
    --dataset_config dataset_config.toml \
    --text_encoder models/flux_klein_4b/text_encoder/model-00001-of-00002.safetensors \
    --model_version klein-4b \
    --batch_size 4
