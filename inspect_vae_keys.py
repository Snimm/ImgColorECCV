import torch
from safetensors.torch import load_file
from musubi_tuner.flux_2.flux2_models import AutoEncoder, AutoEncoderParams

# Load diffusers VAE keys
diffusers_vae_path = "/data/swarnim/ImgColorECCV/models/flux_klein_4b/vae/diffusion_pytorch_model.safetensors"
try:
    diffusers_sd = load_file(diffusers_vae_path)
    print(f"Diffusers keys (first 10): {list(diffusers_sd.keys())[:10]}")
except Exception as e:
    print(f"Error loading diffusers VAE: {e}")

# Instantiate Musubi AutoEncoder to check expected keys
params = AutoEncoderParams()
ae = AutoEncoder(params)
musubi_keys = list(ae.state_dict().keys())
print(f"Musubi keys (first 10): {musubi_keys[:10]}")

# Check for obvious mismatches
if list(diffusers_sd.keys())[0] == musubi_keys[0]:
    print("Keys match exactly!")
else:
    print("Keys differ. Need mapping.")
