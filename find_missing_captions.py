import json
import os
import glob
import argparse

def find_missing(image_dir, jsonl_file, output_list):
    # Load existing
    existing = set()
    if os.path.exists(jsonl_file):
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    existing.add(json.loads(line)['file_name'])
                except:
                    pass
    
    # Check disk
    all_files = glob.glob(os.path.join(image_dir, "*.JPEG")) + \
                glob.glob(os.path.join(image_dir, "*.jpg")) + \
                glob.glob(os.path.join(image_dir, "*.png"))
    
    missing = []
    for fpath in all_files:
        if os.path.basename(fpath) not in existing:
            missing.append(fpath)
            
    print(f"Found {len(missing)} missing images out of {len(all_files)}")
    
    with open(output_list, 'w') as f:
        for m in missing:
            f.write(m + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    find_missing(args.dir, args.jsonl, args.out)
