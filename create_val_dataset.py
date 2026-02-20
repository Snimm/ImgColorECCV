import os
import glob
import json
import random

def create_val_dataset(data_dir, output_file, num_sample_per_class=1):
    val_dir = os.path.join(data_dir, "val")
    if not os.path.exists(val_dir):
        print(f"Validation directory not found: {val_dir}")
        return

    json_lines = []
    classes = glob.glob(os.path.join(val_dir, "*"))
    print(f"Found {len(classes)} classes")
    
    # Sort for deterministic behavior
    classes.sort()
    
    # We want a subset of ~200 images for fast evaluation
    # There are 1000 classes. Let's pick 1 image from every 5th class.
    selected_classes = classes[::5] # 200 classes
    
    count = 0
    for cls_path in selected_classes:
        images = glob.glob(os.path.join(cls_path, "*.JPEG"))
        if not images:
            continue
        
        # Pick just one image
        img_path = images[0]
        
        # Caption is just the class name or generic for colorization
        # For colorization, the caption describes the content. 
        # Since we don't have BLIP running here, we can use a placeholder
        # OR better: use the filename or class code.
        # Actually, for FID we just need real images and generated images.
        # The prompt for generation matters. 
        # We can use "a photo of a [class_name]" if we had class names mapping.
        # Or just use a generic prompt if the model is unconditional or we use "high quality photo"
        # BUT this model is text-to-image (or image-to-image with control).
        # We need prompts.
        # Let's assume we can get prompts from BLIP later or use empty prompt if supported.
        # Wait, previous conversion used "arafed fish...".
        # We should use BLIP to caption these valid images if we want good results.
        # OR we can just use "a high quality photo" as a baseline prompt.
        
        # Let's check if we have captions. We likely don't for val set.
        # I'll use a placeholder "a photo" for now, or "" if possible.
        
        # Construct JSON entry
        entry = {
            "image_path": img_path,
            "prompt": "a high quality color photo", # Placeholder
            "control_image_path": img_path # This will be processed to grayscale in inference
        }
        json_lines.append(json.dumps(entry))
        count += 1
        
    with open(output_file, 'w') as f:
        for line in json_lines:
            f.write(line + "\n")
            
    print(f"Created {output_file} with {count} images.")

if __name__ == "__main__":
    create_val_dataset("/data/swarnim/ImgColorECCV/imagenet-mini", "val_dataset.jsonl")
