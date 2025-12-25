
import torch
import os
from hspmn_v2_1 import HSPMNBlock
from utils_v2_1 import HSPMNConfig

def check_model(filename):
    print(f"Checking {filename}...")
    if not os.path.exists(filename):
        print(f"❌ File {filename} not found.")
        return

    try:
        checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
        print(f"✅ Loaded {filename}")
        
        if 'model' in checkpoint:
            print(f"   - Contains model state_dict with {len(checkpoint['model'])} keys")
        else:
            print("   ❌ Missing 'model' key in checkpoint")
            
        if 'config' in checkpoint:
            print(f"   - Config: {checkpoint['config']}")
        
        if 'step' in checkpoint:
            print(f"   - Step: {checkpoint['step']}")
            
    except Exception as e:
        print(f"❌ Failed to load {filename}: {e}")

if __name__ == "__main__":
    print("--- Verifying Saved Models ---")
    check_model("checkpoint_step_5.pt")
    check_model("checkpoint_step_10.pt")
    check_model("best_model.pt")
