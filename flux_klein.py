#!/usr/bin/env python3
"""
FLUX.2-klein-4B - Official Implementation
Using Flux2KleinPipeline from diffusers 0.37.0.dev0
"""

import torch
from diffusers import Flux2KleinPipeline
import argparse
import os
from datetime import datetime

from PIL import Image

print("=" * 60)
print("FLUX.2-klein-4B - Official Pipeline")
print("=" * 60)

print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nLoading FLUX.2-klein-4B...")

# Load using the official Flux2KleinPipeline
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16
)

print("Moving to GPU...")
pipe = pipe.to("cuda")

print("✓ Ready to generate!")

def generate_image(prompt, image=None, output_path=None, height=1024, width=1024, steps=4, guidance=1.0, seed=None):
    """
    Generate an image with FLUX.2-klein-4B (Supports Text-to-Image and Image-to-Image)
    
    Args:
        prompt: Text description
        image: Optional input image for conditioning (Image-to-Image)
        output_path: Output file path
        height: Image height
        width: Image width
        steps: Number of inference steps (4-8 recommended)
        guidance: Guidance scale (1.0 recommended for klein)
        seed: Random seed for reproducibility
    """
    print(f"\nGenerating image...")
    print(f"Prompt: {prompt}")
    if image:
        print(f"Conditioned on input image: {image}")
    else:
        print(f"Size: {width}x{height}")
    print(f"Steps: {steps}, Guidance: {guidance}")
    
    generator = None
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(seed)
        print(f"Seed: {seed}")
    
    # Load image if path is provided
    conditioning_image = None
    if image:
        conditioning_image = Image.open(image).convert("RGB")
    
    # Generate using the official pipeline
    result = pipe(
        prompt=prompt,
        image=conditioning_image,
        height=height if not conditioning_image else None, # Inherit from image if provided
        width=width if not conditioning_image else None,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator
    )
    
    image_out = result.images[0]
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "flux_klein_img2img" if image else "flux_klein"
        output_path = f"{prefix}_{timestamp}.png"
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    image_out.save(output_path)
    print(f"✓ Image saved to: {output_path}")
    
    return image_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLUX.2-klein-4B Image Generation")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--image", type=str, default=None, help="Optional input image for Img2Img")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps (4-8)")
    parser.add_argument("--guidance", type=float, default=1.0, help="Guidance scale (1.0 recommended)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    try:
        generate_image(
            prompt=args.prompt,
            output_path=args.output,
            height=args.height,
            width=args.width,
            steps=args.steps,
            guidance=args.guidance,
            seed=args.seed
        )
        print("\n" + "=" * 60)
        print("✅ SUCCESS! FLUX.2-klein-4B working!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
