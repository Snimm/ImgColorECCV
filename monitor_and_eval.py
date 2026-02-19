import os
import time
import glob
import json
import subprocess
import shutil
import sys
from cleanfid import fid

# Config
OUTPUT_DIR = "/data/swarnim/ImgColorECCV/outputs/flux_klein_4b_colorization"
VAL_DATASET = "/data/swarnim/ImgColorECCV/val_dataset.jsonl"
EVAL_RESULTS_DIR = "/data/swarnim/ImgColorECCV/eval_results_fid"
MODEL_PATH = "/data/swarnim/ImgColorECCV/models/flux_klein_4b/flux-2-klein-4b.safetensors"
VAE_PATH = "/data/swarnim/ImgColorECCV/models/flux_klein_4b/ae.safetensors"
TEXT_ENCODER_PATH = "/data/swarnim/ImgColorECCV/models/flux_klein_4b/text_encoder/model-00001-of-00002.safetensors"
TRAIN_PID_FILE = "/data/swarnim/ImgColorECCV/train_pid.txt"

# Evaluation params
IMAGE_SIZE = 512
INFER_STEPS = 20
GUIDANCE_SCALE = 3.5
PATIENCE = 3  # Early stopping patience

def get_train_pid():
    if os.path.exists(TRAIN_PID_FILE):
        with open(TRAIN_PID_FILE, 'r') as f:
            return int(f.read().strip())
    return None

def kill_training():
    pid = get_train_pid()
    if pid:
        print(f"Stopping training process {pid}...")
        try:
            os.kill(pid, 15) # SIGTERM
            print("Training stopped.")
        except ProcessLookupError:
            print("Training process already gone.")
    else:
        print("Could not find training PID.")

def generate_images(checkpoint_path, output_dir):
    print(f"Generating images for checkpoint {checkpoint_path}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load validation set
    with open(VAL_DATASET, 'r') as f:
        lines = f.readlines()
    
    # We need to process line by line or batch. 
    # flux_2_generate_image.py takes a single prompt and control image.
    # It doesn't seem to support batch processing from file natively based on my previous view.
    # I'll need to wrap it or modify it. 
    # For now, I'll loop in python calling the script. This is slow but robust.
    # optimizations can be done later.
    
    # Since calling subprocess for every image is super slow (loading model every time),
    # I should write a simple inference script that keeps model loaded.
    # But for now to save time, I will assume I can write a batch inference script.
    # Let me actually write a batch inference script `batch_inference.py`.
    return

def batch_inference(checkpoint_path, output_dir):
    # This function constructs a command to run a batch inference script
    # I'll create `batch_inference.py` separately.
    cmd = [
        sys.executable, "batch_inference.py",
        "--checkpoint", checkpoint_path,
        "--output_dir", output_dir,
        "--dataset", VAL_DATASET
    ]
    subprocess.run(cmd, check=True)

def compute_fid(real_dir, fake_dir):
    score = fid.compute_fid(real_dir, fake_dir)
    return score

def main():
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    
    # Create directory for real images (copy from val dataset to a folder)
    real_images_dir = os.path.join(EVAL_RESULTS_DIR, "real_images")
    if not os.path.exists(real_images_dir):
        os.makedirs(real_images_dir, exist_ok=True)
        print("Preparing real images for FID...")
        with open(VAL_DATASET, 'r') as f:
            for line in f:
                entry = json.loads(line)
                src = entry["image_path"]
                dst = os.path.join(real_images_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy(src, dst)
    
    processed_checkpoints = set()
    fid_history = []
    
    print("Monitoring for checkpoints...")
    
    while True:
        # Check for new checkpoints
        checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "*.safetensors"))
        checkpoints.sort(key=os.path.getmtime)
        
        for ckpt in checkpoints:
            if ckpt in processed_checkpoints:
                continue
            
            # Wait for file to stabilize (simple check)
            time.sleep(5) 
            
            print(f"New checkpoint found: {ckpt}")
            
            # Create output dir for this checkpoint
            ckpt_name = os.path.splitext(os.path.basename(ckpt))[0]
            fake_images_dir = os.path.join(EVAL_RESULTS_DIR, ckpt_name)
            
            try:
                # Generate images
                batch_inference(ckpt, fake_images_dir)
                
                # Compute FID
                print("Computing FID...")
                score = compute_fid(real_images_dir, fake_images_dir)
                print(f"FID for {ckpt_name}: {score}")
                
                # Log
                with open(os.path.join(EVAL_RESULTS_DIR, "fid_log.txt"), "a") as f:
                    f.write(f"{ckpt_name},{score}\n")
                
                # Early stopping
                fid_history.append(score)
                if len(fid_history) > PATIENCE:
                    recent = fid_history[-PATIENCE:]
                    # If consistently increasing
                    if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
                        print(f"Early stopping triggered! FID increased for {PATIENCE} epochs.")
                        kill_training()
                        sys.exit(0)
                        
                processed_checkpoints.add(ckpt)
                
            except Exception as e:
                print(f"Error processing {ckpt}: {e}")
        
        time.sleep(60) # Check every minute

if __name__ == "__main__":
    main()
