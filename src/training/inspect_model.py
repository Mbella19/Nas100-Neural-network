import torch
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

def inspect_checkpoint():
    path = os.path.join(project_root, 'models/analyst/best_model.pth')
    if not os.path.exists(path):
        path = os.path.join(project_root, 'models/analyst/best.pt')
        
    if not os.path.exists(path):
        print(f"Checkpoint not found at {path}")
        return

    print(f"Inspecting checkpoint: {path}")
    try:
        checkpoint = torch.load(path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Check encoder_15m.input_projection.weight
        key = 'encoder_15m.input_projection.weight'
        if key in state_dict:
            weight = state_dict[key]
            print(f"Found {key} with shape: {weight.shape}")
            print(f"Input features expected: {weight.shape[1]}")
        else:
            print(f"Key {key} not found in state_dict")
            print("Keys found:", list(state_dict.keys())[:10])
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    inspect_checkpoint()
