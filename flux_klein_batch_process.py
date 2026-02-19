#!/usr/bin/env python3
"""
FLUX.2-klein-4B - Batch Image Processing
Iterates through a directory, applies "colorize this image" prompt using Image-to-Image.
"""

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
import os
import glob
from tqdm import tqdm
import argparse
from datetime import datetime

# Configuration
INPUT_DIR = "/data/swarnim/ImgColorECCV/dataset/control_images"
CAPTIONS_DIR = "/data/swarnim/ImgColorECCV/dataset/captions"
OUTPUT_BASE_DIR = "/data/swarnim/ImgColorECCV/dataset/output/colorized_results"
DEFAULT_PROMPT = "colorize this image"

def batch_process_images(input_dir, captions_dir, output_dir, prompt, height=1024, width=1024, steps=4, guidance=1.0, seed=42, shard_idx=0, num_shards=1):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"FLUX.2-klein-4B - Batch Processing")
    print(f"Input Directory: {input_dir}")
    print(f"Captions Directory: {captions_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Prompt Prefix: {prompt}")
    print(f"Shard: {shard_idx + 1}/{num_shards}")
    print("=" * 60)

    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper()))) # Case insensitive check
    
    image_files = sorted(list(set(image_files))) # Unique and sorted
    
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return

    total_images = len(image_files)
    if num_shards > 1:
        image_files = image_files[shard_idx::num_shards]
    print(f"Found {total_images} images total.")
    print(f"Processing {len(image_files)} images for this shard.")

    # Load Model
    print("\nLoading FLUX.2-klein-4B model...")
    try:
        pipe = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B",
            torch_dtype=torch.bfloat16
        )
        pipe = pipe.to("cuda")
        print("✓ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Generator for reproducibility
    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None

    # Process Loop
    success_count = 0
    failure_count = 0

    for img_path in tqdm(image_files, desc="Processing Images"):
        try:
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, filename)
            base_name, _ = os.path.splitext(filename)
            caption_path = os.path.join(captions_dir, f"{base_name}.txt")
            caption_text = None

            if os.path.exists(caption_path):
                with open(caption_path, "r", encoding="utf-8") as f:
                    caption_text = f.read().strip()

            if caption_text:
                full_prompt = f"{prompt}, {caption_text}"
            else:
                full_prompt = prompt
            
            # Skip if already exists (optional, could add a flag to force overwrite)
            if os.path.exists(output_path):
                 # print(f"Skipping {filename}, already exists.")
                 pass 
                 # Continuing to overwrite for now as is typical for scripts unless specified otherwise

            # Load image
            input_image = Image.open(img_path).convert("RGB")

            # Generate
            result = pipe(
                image=input_image,
                prompt=full_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            )
            
            generated_image = result.images[0]
            
            # Save
            generated_image.save(output_path)
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ Error processing {filename}: {e}")
            failure_count += 1
            continue

    print("\n" + "=" * 60)
    print(f"Processing Complete!")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {failure_count}")
    print(f"Results saved in: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Colorize Images with Flux Klein")
    parser.add_argument("--input_dir", type=str, default=INPUT_DIR, help="Directory containing input images")
    parser.add_argument("--captions_dir", type=str, default=CAPTIONS_DIR, help="Directory containing caption .txt files")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_BASE_DIR, help="Directory to save output images")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt to use for generation")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--shard_idx", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    
    args = parser.parse_args()
    
    batch_process_images(
        input_dir=args.input_dir,
        captions_dir=args.captions_dir,
        output_dir=args.output_dir,
        prompt=args.prompt,
        steps=args.steps,
        seed=args.seed,
        shard_idx=args.shard_idx,
        num_shards=args.num_shards
    )
