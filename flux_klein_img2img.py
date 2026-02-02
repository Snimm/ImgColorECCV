#!/usr/bin/env python3
"""
FLUX.2-klein-4B - Image-to-Image / Image-Conditioned Generation
Using Flux2KleinPipeline's 'image' parameter
"""

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
import argparse
import os
from datetime import datetime

print("=" * 60)
print("FLUX.2-klein-4B - Image-Conditioned Gen")
print("=" * 60)

def generate_img2img(image_path, prompt, output_path=None, height=1024, width=1024, steps=4, guidance=1.0, seed=None):
    print(f"\nLoading input image: {image_path}")
    input_image = Image.open(image_path).convert("RGB")
    
    print("\nLoading FLUX.2-klein-4B...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        torch_dtype=torch.bfloat16
    )
    print("Moving to GPU...")
    pipe = pipe.to("cuda")

    print(f"\nGenerating image conditioned on input...")
    print(f"Prompt: {prompt}")
    
    generator = None
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(seed)
    
    # Pass the loaded image to the 'image' argument
    result = pipe(
        image=input_image,
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator
    )
    
    image = result.images[0]
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"flux_klein_img2img_{timestamp}.png"
    
    image.save(output_path)
    print(f"✓ Image saved to: {output_path}")
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLUX.2-klein-4B Img2Img")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    try:
        generate_img2img(
            image_path=args.image,
            prompt=args.prompt,
            output_path=args.output,
            height=args.height,
            width=args.width,
            steps=args.steps,
            seed=args.seed
        )
        print("\n✅ SUCCESS!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
