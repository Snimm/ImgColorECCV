import torch
from safetensors.torch import load_file
import sys

def count_lora_params(lora_path):
    try:
        state_dict = load_file(lora_path)
        total_params = 0
        for key, tensor in state_dict.items():
            total_params += tensor.numel()
        
        print(f"Total LoRA Parameters: {total_params:,}")
        print(f"Total LoRA Parameters (Million): {total_params / 1e6:.2f}M")
    except Exception as e:
        print(f"Error loading LoRA: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_lora_params.py <path_to_lora>")
        sys.exit(1)
    
    count_lora_params(sys.argv[1])
