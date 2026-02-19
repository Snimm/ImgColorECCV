import os
import glob
import json
import argparse
from tqdm import tqdm

def create_jsonl(image_dir, output_file, caption_extension=".txt"):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.WEBP']
    image_files = []
    print(f"Searching for images in {image_dir}...")
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
    
    image_files = sorted(list(set(image_files)))
    print(f"Found {len(image_files)} images.")
    
    data = []
    for img_path in tqdm(image_files, desc="Processing"):
        base_name, _ = os.path.splitext(img_path)
        caption_path = f"{base_name}{caption_extension}"
        
        caption = ""
        if os.path.exists(caption_path):
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        else:
            print(f"Warning: No caption found for {img_path}")
            continue # Skip images without captions or use empty string? Better to skip if training expects captions.
        
        # Absolute path is safer
        abs_img_path = os.path.abspath(img_path)
        
        entry = {
            "image_path": abs_img_path,
            "caption": caption
        }
        data.append(entry)
        
    print(f"Writing {len(data)} entries to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create JSONL dataset for musubi-tuner")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    args = parser.parse_args()
    
    create_jsonl(args.image_dir, args.output_file)
