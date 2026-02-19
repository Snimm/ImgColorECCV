#!/bin/bash
export PYTHONPATH=""
# Isolate from system python packages
export PYTHONNOUSERSITE=1

# Use the virtual environment python created by uv
PYTHON_EXEC="$(pwd)/.uv_venv/bin/python"

# Check if accelerate config exists, otherwise use default
if [ ! -f "accelerate_config.yaml" ]; then
    echo "Creating default accelerate config..."
    $PYTHON_EXEC -m accelerate.commands.accelerate_cli config default --config_file accelerate_config.yaml
fi

# Check for existing checkpoints to resume
RESUME_ARGS=""
LATEST_CHECKPOINT=$(ls -d /data/swarnim/ImgColorECCV/outputs/flux_klein_4b_colorization/flux_klein_4b_colorization-step* 2>/dev/null | sort -V | tail -n 1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Found checkpoint: $LATEST_CHECKPOINT. Resuming training..."
    RESUME_ARGS="--resume $LATEST_CHECKPOINT"
fi

# Launch training
echo "Launching training with $PYTHON_EXEC"
$PYTHON_EXEC -m accelerate.commands.accelerate_cli launch --num_processes=4 --multi_gpu --mixed_precision=bf16 \
    musubi-tuner/flux_2_train_network.py \
    --config_file /data/swarnim/ImgColorECCV/train_config.toml \
    --dit /data/swarnim/ImgColorECCV/models/flux_klein_4b/flux-2-klein-4b.safetensors \
    --vae /data/swarnim/ImgColorECCV/models/flux_klein_4b/ae.safetensors \
    --text_encoder /data/swarnim/ImgColorECCV/models/flux_klein_4b/text_encoder/model-00001-of-00002.safetensors \
    --network_module musubi_tuner.networks.lora \
    --network_dim 16 \
    --network_alpha 16 \
    $RESUME_ARGS &

PID=$!
echo $PID > train_pid.txt
echo "Training started with PID $PID"
wait $PID
