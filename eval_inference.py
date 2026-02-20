import torch
import cv2
import sys
import os
import argparse
import subprocess
from PIL import Image
import numpy as np

# Sample image from user
SAMPLE_IMAGE = "/data/swarnim/ImgColorECCV/imagenet-mini/train/n01440764/n01440764_10043.JPEG"
OUTPUT_DIR = "eval_results"
MODEL_PATH = "/data/swarnim/ImgColorECCV/models/flux_klein_4b/flux-2-klein-4b.safetensors"
VAE_PATH = "/data/swarnim/ImgColorECCV/models/flux_klein_4b/ae.safetensors"
TEXT_ENCODER_PATH = "/data/swarnim/ImgColorECCV/models/flux_klein_4b/text_encoder/model-00001-of-00002.safetensors"
LORA_PATH = "/data/swarnim/ImgColorECCV/outputs/flux_klein_4b_colorization/flux_klein_4b_colorization.safetensors"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Prepare grayscale control image
    img = cv2.imread(SAMPLE_IMAGE)
    if img is None:
        print(f"Error: Could not read image {SAMPLE_IMAGE}")
        sys.exit(1)
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    control_path = os.path.join(OUTPUT_DIR, "control.jpg")
    cv2.imwrite(control_path, gray)
    print(f"Saved grayscale control image to {control_path}")
    
    # 2. Run inference
    cmd = [
        sys.executable, "musubi-tuner/src/musubi_tuner/flux_2_generate_image.py",
        "--model_version", "klein-4b",
        "--dit", MODEL_PATH,
        "--vae", VAE_PATH,
        "--text_encoder", TEXT_ENCODER_PATH,
        "--lora_weight", LORA_PATH,
        "--lora_multiplier", "1.0",
        "--control_image_path", control_path,
        "--prompt", "arafed fish in a man ' s hand is being held by a boy",
        "--save_path", OUTPUT_DIR,
        "--image_size", "512", "512",
        "--infer_steps", "20",
        "--guidance_scale", "3.5",
        "--seed", "42",
        "--device", "cuda:0"
    ]
    
    print("Running inference...")
    subprocess.run(cmd, check=True)
    print(f"Inference complete. Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
