import torch
from PIL import Image
import json
import os
import argparse
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_list", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print(f"Loading BLIP2 on {args.device}...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    ).to(args.device)

    with open(args.input_list, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(paths)} images...")
    
    with open(args.output_jsonl, 'a') as f:
        for path in tqdm(paths):
            try:
                img = Image.open(path).convert("RGB")
                inputs = processor(images=img, return_tensors="pt").to(args.device, torch.float16)
                with torch.no_grad():
                    generated_ids = model.generate(**inputs)
                    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
                f.write(json.dumps({"file_name": os.path.basename(path), "caption": caption}) + "\n")
            except Exception as e:
                print(f"Error {path}: {e}")

if __name__ == "__main__":
    main()
