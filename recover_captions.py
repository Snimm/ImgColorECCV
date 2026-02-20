import re
import json
import os
import argparse

def extract_captions(log_file, output_jsonl):
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return

    # To handle duplicates from multiple runs in the same log
    captions = {} 
    
    current_file = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Pattern: INFO:__main__:[1/5000] Processing ILSVRC2012_val_00000001.JPEG
            proc_match = re.search(r"Processing ([\w\.\-]+)", line)
            if proc_match:
                current_file = proc_match.group(1)
                continue
            
            # Pattern: INFO:__main__:  Caption: a large snake laying on the sand
            cap_match = re.search(r"Caption: (.+)", line)
            if cap_match and current_file:
                captions[current_file] = cap_match.group(1).strip()
                current_file = None

    print(f"Extracted {len(captions)} captions from {log_file}")
    
    # Write to JSONL
    with open(output_jsonl, 'w') as f:
        for fname, caption in sorted(captions.items()):
            f.write(json.dumps({"file_name": fname, "caption": caption}) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    extract_captions(args.log, args.out)
