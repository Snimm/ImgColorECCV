import os
import glob
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

def caption_images(image_dir, caption_extension=".txt", device="cuda", batch_size=4, model_name="Salesforce/blip-image-captioning-large"):
    # Load BLIP2 model and processor
    print(f"Loading model: {model_name}")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    # model.to(device) # device_map="auto" handles device placement, so this line is often not needed or can conflict.
    
    # Find images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.WEBP']
    image_files = []
    print(f"Searching for images in {image_dir}...")
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
    
    image_files = sorted(list(set(image_files)))
    print(f"Found {len(image_files)} images.")
    
    # Process in batches
    for i in tqdm(range(0, len(image_files), batch_size), desc="Captioning"):
        batch_files = image_files[i:i+batch_size]
        images = []
        valid_files = []
        
        for img_path in batch_files:
            try:
                # Check if caption already exists
                base_name, _ = os.path.splitext(img_path)
                caption_path = f"{base_name}{caption_extension}"
                if os.path.exists(caption_path):
                    continue
                
                image = Image.open(img_path).convert('RGB')
                images.append(image)
                valid_files.append(img_path)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        if not images:
            continue
            
        inputs = processor(images=images, return_tensors="pt").to(device, torch.float16)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
        for img_path, caption in zip(valid_files, generated_text):
            base_name, _ = os.path.splitext(img_path)
            caption_path = f"{base_name}{caption_extension}"
            with open(caption_path, "w") as f:
                f.write(caption.strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Caption images using BLIP2")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to directory containing images")
    parser.add_argument("--model_name", type=str, default="Salesforce/blip-image-captioning-large", help="HuggingFace model name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    args = parser.parse_args()
    
    caption_images(args.image_dir, model_name=args.model_name, batch_size=args.batch_size)
