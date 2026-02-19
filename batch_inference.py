import argparse
import os
import sys
import json
import torch
import cv2
import numpy as np
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import musubi-tuner modules
sys.path.append(os.getcwd())
try:
    from musubi_tuner import flux_2_generate_image as flux2_generate_image
    from musubi_tuner.flux_2 import flux2_utils
    from musubi_tuner.flux_2 import flux2_models
    from musubi_tuner.utils.device_utils import clean_memory_on_device
except ImportError:
    sys.path.append("musubi-tuner/src")
    from musubi_tuner import flux_2_generate_image as flux2_generate_image
    from musubi_tuner.flux_2 import flux2_utils
    from musubi_tuner.flux_2 import flux2_models
    from musubi_tuner.utils.device_utils import clean_memory_on_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    
    # Mock args for flux2_generate_image functions
    class GenArgs:
        dit = "/data/swarnim/ImgColorECCV/models/flux_klein_4b/flux-2-klein-4b.safetensors"
        vae = "/data/swarnim/ImgColorECCV/models/flux_klein_4b/ae.safetensors"
        text_encoder = "/data/swarnim/ImgColorECCV/models/flux_klein_4b/text_encoder/model-00001-of-00002.safetensors"
        model_version = "klein-4b"
        device = "cuda:0"
        lora_weight = [args.checkpoint]
        lora_multiplier = [1.0]
        include_patterns = None
        exclude_patterns = None
        lycoris = False
        blocks_to_swap = 0
        fp8 = False
        fp8_scaled = False
        attn_mode = "sdpa" # or torch
        disable_numpy_memmap = False
        save_merged_model = None
        compile = False
        
        # Generation params
        image_size = [512, 512]
        infer_steps = 20
        guidance_scale = 3.5
        seed = 42
        fp8_text_encoder = False
        prompt = None # Set per image
        negative_prompt = None
        control_image_path = None # Set per image
        no_resize_control = False
        flow_shift = None
        output_type = "images"
        no_metadata = True
        latent_path = None
        
    gen_args = GenArgs()
    device = torch.device(gen_args.device)
    
    # Load Models (DiT, VAE, Text Encoder)
    logger.info("Loading models...")
    
    # Load VAE
    # Note: load_ae takes dtype args.vae_dtype which we don't have in GenArgs, but load_ae handles defaults?
    # Actually flux2_utils.load_ae signature: (vae_path, dtype, device, disable_mmap)
    vae = flux2_utils.load_ae(gen_args.vae, dtype=torch.bfloat16, device=device)
    
    # Load DiT
    # flux2_generate_image.load_dit_model(args, device, dit_weight_dtype)
    dit = flux2_generate_image.load_dit_model(gen_args, device, torch.bfloat16)

    # Load Text Encoder
    model_version_info = flux2_utils.FLUX2_MODEL_INFO[gen_args.model_version]
    text_embedder = flux2_utils.load_text_embedder(
        model_version_info, gen_args.text_encoder, dtype=torch.bfloat16, device=device, disable_mmap=True
    )
    
    # Pack models into shared_models dict for generate function
    shared_models = {
        "model": dit,
        "text_embedder": text_embedder,
        "conds_cache": {}
    }
    
    # Settings
    gen_settings = flux2_generate_image.GenerationSettings(device, torch.bfloat16)
    
    # Process Dataset
    with open(args.dataset, 'r') as f:
        lines = f.readlines()
        
    logger.info(f"Generating {len(lines)} images...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for i, line in enumerate(lines):
        try:
            entry = json.loads(line)
            prompt = entry["prompt"]
            control_img_path = entry["image_path"] # In dataset creation we used same path. 
            # We need to ensure it exists and matches what we expect
            
            # Create a grayscale version temp or pass it directly? 
            # The model is trained on grayscale input. The inference script handles control image preprocessing?
            # flux2_utils.preprocess_control_image handles it.
            # But we want to ensure we feed a grayscale image as control if that's what we trained on.
            # Our training data loader converts to grayscale on fly.
            # So here we should also convert.
            
            # Ideally we save a temp grayscale image.
            # Or reliance on flux2_utils? No, flux2_utils just loads the image.
            # We must provide the grayscale image path.
            
            # Let's verify what `eval_inference.py` did. It converted to grayscale.
            # So we must do the same.
            
            # Save temp grayscale
            temp_control_path = os.path.join(args.output_dir, f"temp_control_{i}.jpg")
            img = cv2.imread(control_img_path)
            if img is None:
                logger.warning(f"Could not read {control_img_path}, skipping")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(temp_control_path, gray)
            
            # Update args for this item
            gen_args.prompt = prompt
            gen_args.control_image_path = [temp_control_path]
            gen_args.seed = 42 + i # Vary seed? Or keep fixed? Fixed seed per image is good for determinism.
            
            # Generate
            # generate() returns (vae, latents) if output_type is latent, or images if not?
            # Wait, looking at generate(), it calls `decode_latent` if output_type is images.
            # It returns (vae, latent)
            
            # Actually, `flux_2_generate_image.py` main loop handles decoding.
            # Let's see `main` loop in `flux_2_generate_image.py` lines 1120+
            # It calls `generate`, gets `returned_vae` and `latent`.
            # Then if `returned_vae` is not None, it decodes.
            
            # Call generate
            returned_vae, latent = flux2_generate_image.generate(gen_args, gen_settings, shared_models=shared_models)
            
            # Decode
            # We can use the loaded `vae` if returned_vae is None (it might be None if shared_models was passed?)
            # Actually generate() returns None for vae if it didn't load one locally? 
            # Let's check generate() implementation in previous view (Step 1657).
            # It returns `vae_instance_for_return` which is None if `vae` was passed in shared_models or args?
            # Wait, `generate` takes `args`. It doesn't take `vae` directly? 
            # Ah, `prepare_i2v_inputs` takes `ae`. `generate` calls `prepare_i2v_inputs`.
            # But `generate` loads `ae` if not passed? 
            # Line 648 in generate(): `ae = flux2_utils.load_ae(...)` if not in shared_models?
            # Actually `generate` signature in my view:
            # def generate(args, gen_settings, shared_models=None, ...)
            # It seems it doesn't take `ae` in shared_models?
            # Let's check `generate` body again.
            
            # I can't see the full body of `generate` in the previous view. 
            # But I can modify `generate` arguments or rely on `flux2_generate_image` doing the right thing.
            # Since I passed `shared_models` with `model` and `text_embedder`, but NO `ae` key in `shared_models` dict?
            # My code above: `shared_models = {"model": dit, "text_embedder": text_embedder}`.
            # Does `generate` look for `ae` in `shared_models`?
            # Most likely not, usually VAE is loaded inside.
            
            # Optimize: Modify `flux_2_generate_image.py` to accept `ae` in shared_models if it doesn't already?
            # Or just let it load VAE every time? Loading VAE is fast compared to DiT? 
            # VAE is small (few hundred MB). It's fine.
            # BUT, we want to avoid reloading if possible.
            
            # Actually, `prepare_i2v_inputs` takes `ae`.
            # `generate` calls `prepare_i2v_inputs`.
            # `generate` must obtain `ae` somewhere.
            
            # If I look at `musubi_tuner/flux_2_generate_image.py`, I can check if I can pass `ae`.
            # I'll Assume standard behavior: it loads `vae` inside.
            # To avoid reloading, I might need to hack it or just accept the VAE load overhead (small).
            
            # Let's just call `generate`.
            
            # Decode using my pre-loaded VAE to save time if `generate` returns latent.
            # `generate` returns `(returned_vae, latent)`.
            # If `returned_vae` is available, use it. If not, use my local `vae`.
            
            if latent is not None:
                # Decode
                image = flux2_generate_image.decode_latent(vae, latent, device)
                
                # Save
                save_path = os.path.join(args.output_dir, f"image_{i:04d}.png")
                # image is tensor or PIL? decode_latent returns tensor? 
                # decode_latent returns (C,H,W) tensor in range [-1, 1]?
                
                # Let's check decode_latent in view.
                # Line 416: pixels = ae.decode(...)
                # Line 421: returns pixels[0] (remove batch)
                # It returns tensor.
                
                # Convert to PIL
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(1, 2, 0).numpy()
                image = (image * 255).astype(np.uint8)
                Image.fromarray(image).save(save_path)
            
            # Clean up temp file
            if os.path.exists(temp_control_path):
                os.remove(temp_control_path)
                
        except Exception as e:
            logger.error(f"Failed to generate {i}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
