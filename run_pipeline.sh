#!/bin/bash
set -e

# activate venv
source .uv_venv/bin/activate

# 1. Run Caching
echo "Starting Caching..."
./cache_latents.sh

# 2. Run Training
echo "Caching complete. Starting Training..."
./start_training.sh
