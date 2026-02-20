import argparse
import os
import sys
import glob
import cv2
import numpy as np
import torch
from PIL import Image
import logging
import json
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import musubi-tuner modules with alias fix
sys.path.append(os.getcwd())
try:
    from musubi_tuner import flux_2_generate_image as flux2_generate_image
    from musubi_tuner.flux_2 import flux2_utils
    from musubi_tuner.flux_2 import flux2_models
except ImportError:
    sys.path.append("musubi-tuner/src")
    from musubi_tuner import flux_2_generate_image as flux2_generate_image
    from musubi_tuner.flux_2 import flux2_utils
    from musubi_tuner.flux_2 import flux2_models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--image_dir", type=str, default="/data/swarnim/DATA/swarnim/imagenet5k", help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="/data/swarnim/ImgColorECCV/eval_results_imagenet5k_ar_fix", help="Directory to save results")
    parser.add_argument("--limit", type=int, default=10, help="Number of images to process")
    parser.add_argument("--flux_device", type=str, default="cuda:0", help="Device for Flux model")
    parser.add_argument("--blip_device", type=str, default="cuda:1", help="Device for BLIP model")
    parser.add_argument("--caption_file", type=str, default=None, help="Path to JSONL file containing captions")
    parser.add_argument("--grayscale_mode", type=str, default="standard", choices=["standard", "ortho"], help="Grayscale conversion mode: 'standard' (OpenCV) or 'ortho' ((B+G)/2)")
    args = parser.parse_args()

    # Settings for devices
    flux_device = args.flux_device
    blip_device = args.blip_device
    logger.info(f"Flux Device: {flux_device}, BLIP Device: {blip_device}")

    # 1. Load Captions if provided
    caption_map = {}
    if args.caption_file:
        logger.info(f"Loading captions from {args.caption_file}...")
        with open(args.caption_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    caption_map[data["file_name"]] = data["caption"]
                except Exception as e:
                    logger.warning(f"Failed to parse line in caption file: {e}")
        logger.info(f"Loaded {len(caption_map)} captions.")

    # 2. Load BLIP2 Model (Conditional)
    processor = None
    blip_model = None
    if not args.caption_file:
        logger.info("Loading BLIP2 model...")
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        # Load in float16 to save memory
        blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )
        blip_model.to(blip_device)
        logger.info("BLIP2 loaded.")
    else:
        logger.info("Skipping BLIP2 loading (using caption file).")

    # 3. Load Flux Model Configuration
    class GenArgs:
        dit = "/data/swarnim/ImgColorECCV/models/flux_klein_4b/flux-2-klein-4b.safetensors"
        vae = "/data/swarnim/ImgColorECCV/models/flux_klein_4b/ae.safetensors"
        text_encoder = "/data/swarnim/ImgColorECCV/models/flux_klein_4b/text_encoder/model-00001-of-00002.safetensors"
        model_version = "klein-4b"
        device = flux_device
        lora_weight = [args.checkpoint]
        lora_multiplier = [1.0]
        # Defaults
        include_patterns = None
        exclude_patterns = None
        lycoris = False
        blocks_to_swap = 0
        fp8 = False
        fp8_scaled = False
        attn_mode = "torch" # Safe fallback
        disable_numpy_memmap = False
        save_merged_model = None
        compile = False
        # Gen params
        image_size = [512, 512] # Will be updated per image
        infer_steps = 20
        guidance_scale = 3.5
        embedded_cfg_scale = 3.5 
        seed = 42
        fp8_text_encoder = False
        prompt = "" # Set per image
        negative_prompt = None
        control_image_path = None # Set per image
        no_resize_control = False
        flow_shift = None
        output_type = "images"
        no_metadata = True
        latent_path = None

    gen_args = GenArgs()
    device = torch.device(gen_args.device)

    # Load Flux Models (Load once)
    logger.info("Loading Flux models...")
    vae = flux2_utils.load_ae(gen_args.vae, dtype=torch.bfloat16, device=device)
    dit = flux2_generate_image.load_dit_model(gen_args, device, torch.bfloat16)
    
    model_version_info = flux2_utils.FLUX2_MODEL_INFO[gen_args.model_version]
    text_embedder = flux2_utils.load_text_embedder(
        model_version_info, gen_args.text_encoder, dtype=torch.bfloat16, device=device, disable_mmap=True
    )

    shared_models = {
        "model": dit,
        "text_embedder": text_embedder,
        "conds_cache": {}
    }
    gen_settings = flux2_generate_image.GenerationSettings(device, torch.bfloat16)

    # Gather Images
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.JPEG")) + glob.glob(os.path.join(args.image_dir, "*.jpg")) + glob.glob(os.path.join(args.image_dir, "*.png")))
    logger.info(f"Found {len(image_paths)} images in {args.image_dir}")
    
    if args.limit > 0:
        image_paths = image_paths[:args.limit]
        logger.info(f"Processing first {len(image_paths)} images")

    os.makedirs(args.output_dir, exist_ok=True)

    # Inference Loop
    for i, img_path in enumerate(image_paths):
        basename = os.path.basename(img_path)
        save_path = os.path.join(args.output_dir, f"{os.path.splitext(basename)[0]}_colorized.jpg")
        
        if os.path.exists(save_path):
            logger.info(f"[{i+1}/{len(image_paths)}] Skipping {basename} (Already exists)")
            continue

        logger.info(f"[{i+1}/{len(image_paths)}] Processing {basename}")
        
        try:
            # 1. Get Caption
            generated_text = None
            if basename in caption_map:
                generated_text = caption_map[basename]
                logger.info(f"  Using pre-defined caption: {generated_text}")
            elif blip_model is not None:
                # 1. Read and Resize for BLIP (only if we need BLIP)
                img = cv2.imread(img_path)
                if img is None: continue
                orig_h, orig_w = img.shape[:2]
                max_dim = 512
                scale = max_dim / max(orig_h, orig_w)
                target_h = int(round(orig_h * scale / 16) * 16)
                target_w = int(round(orig_w * scale / 16) * 16)
                img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
                
                inputs = processor(images=pil_image, return_tensors="pt").to(blip_device, torch.float16)
                with torch.no_grad():
                    generated_ids = blip_model.generate(**inputs)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                logger.info(f"  BLIP2 Caption: {generated_text}")
            else:
                logger.warning(f"  No caption found for {basename} and BLIP2 not loaded. Skipping.")
                continue

            # 2. Prepare Images for Flux
            # We need the resized image and grayscale version regardless of where caption came from
            img = cv2.imread(img_path)
            if img is None: continue
            orig_h, orig_w = img.shape[:2]
            max_dim = 512
            scale = max_dim / max(orig_h, orig_w)
            target_h = int(round(orig_h * scale / 16) * 16)
            target_w = int(round(orig_w * scale / 16) * 16)
            target_h, target_w = max(16, target_h), max(16, target_w)
            
            img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            
            # Select Grayscale Mode
            if args.grayscale_mode == "ortho":
                # Orthochromatic: (B + G) / 2
                # OpenCV uses BGR order
                b = img_resized[:, :, 0].astype(np.float32)
                g = img_resized[:, :, 1].astype(np.float32)
                gray = ((b + g) / 2.0).astype(np.uint8)
                logger.info("  Using Orthochromatic Grayscale ((B+G)/2)")
            else:
                gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            
            temp_control_path = os.path.join(args.output_dir, f"temp_control_{i}.jpg")
            cv2.imwrite(temp_control_path, gray)

            # 3. Setup Args for Flux
            gen_args.image_size = [target_h, target_w] 
            gen_args.control_image_path = [temp_control_path]
            gen_args.prompt = generated_text
            gen_args.seed = 42 + i 
            
            # 4. Generate with Flux
            returned_vae, latent = flux2_generate_image.generate(gen_args, gen_settings, shared_models=shared_models)
            
            # 5. Decode and Save
            if latent is not None:
                decoded = flux2_generate_image.decode_latent(vae, latent, device)
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                decoded = decoded.float().cpu().permute(1, 2, 0).numpy()
                decoded = (decoded * 255).astype(np.uint8)
                decoded_img = Image.fromarray(decoded)
                
                if decoded_img.size != (target_w, target_h):
                     decoded_img = decoded_img.resize((target_w, target_h))

                save_path = os.path.join(args.output_dir, f"{os.path.splitext(basename)[0]}_colorized.jpg")
                decoded_img.save(save_path)
                logger.info(f"Saved {save_path}")

                # 6. Log Caption to JSONL (if it wasn't already in the input file)
                if not args.caption_file:
                    jsonl_path = os.path.join(args.output_dir, "captions.jsonl")
                    with open(jsonl_path, "a") as f:
                        f.write(json.dumps({"file_name": basename, "caption": generated_text}) + "\n")
            
            if os.path.exists(temp_control_path):
                os.remove(temp_control_path)

        except Exception as e:
            logger.error(f"Error processing {basename}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
