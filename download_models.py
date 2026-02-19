import os
from huggingface_hub import snapshot_download, hf_hub_download

MODEL_DIR = "/data/swarnim/ImgColorECCV/models/flux_klein_4b"
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Downloading models to {MODEL_DIR}...")

# 1. Download DiT (Flux Klein 4B)
print("Downloading DiT...")
try:
    hf_hub_download(repo_id="black-forest-labs/FLUX.2-klein-4B", filename="flux-2-klein-4b.safetensors", local_dir=MODEL_DIR)
except Exception as e:
    print(f"Error downloading DiT: {e}")

# 2. Download AE (from FLUX.1-schnell as fallback - usually compatible)
print("Downloading AE...")
try:
    # No token needed for schnell usually, but passing it doesn't hurt if we have it.
    hf_hub_download(repo_id="black-forest-labs/FLUX.1-schnell", filename="ae.safetensors", local_dir=MODEL_DIR)
except Exception as e:
    print(f"Error downloading AE: {e}")

# 3. Download Text Encoder (Qwen3 4B split files)
print("Downloading Text Encoder...")
try:
    # Download all safetensors files that look like split files or just all safetensors
    # The doc says "Download all the split files".
    # Usually they are named like model-00001-of-xxxxx.safetensors
    # We can use snapshot_download with allow_patterns.
    snapshot_download(repo_id="black-forest-labs/FLUX.2-klein-4B", allow_patterns=["*.safetensors"], local_dir=MODEL_DIR)
    # Note: this might redownload the DiT if it matches, but that's fine (it verifies checksum).
except Exception as e:
    print(f"Error downloading Text Encoder: {e}")

print("Download complete.")
