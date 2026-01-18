"""
Verify collected safety activations.

Usage:
    python verify_activations.py
    python verify_activations.py --activations_dir safety_activations
"""

import argparse
import torch
from pathlib import Path
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Verify collected activations")
    parser.add_argument("--activations_dir", type=str, default="safety_activations",
                       help="Activations directory (default: safety_activations)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    activations_dir = Path(args.activations_dir)
    
    if not activations_dir.exists():
        print(f"Error: Directory {activations_dir} does not exist!")
        return
    
    print("="*70)
    print("ACTIVATION VERIFICATION")
    print("="*70)
    
    # Load metadata
    metadata_path = activations_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print("\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    else:
        print("\nNo metadata found!")
    
    # Check each layer
    print("\n" + "="*70)
    print("Layer-wise Summary:")
    print("="*70)
    
    layer_dirs = sorted([d for d in activations_dir.iterdir() if d.is_dir() and d.name.startswith("layer_")])
    
    for layer_dir in layer_dirs:
        layer_name = layer_dir.name
        print(f"\n{layer_name}:")
        
        # Check harmful activations
        harmful_path = layer_dir / "harmful.pt"
        if harmful_path.exists():
            harmful_acts = torch.load(harmful_path)
            print(f"  harmful.pt: shape={harmful_acts.shape}, dtype={harmful_acts.dtype}")
            print(f"    Mean: {harmful_acts.mean():.4f}, Std: {harmful_acts.std():.4f}")
        else:
            print("  harmful.pt: NOT FOUND")
        
        # Check helpful activations
        helpful_path = layer_dir / "helpful.pt"
        if helpful_path.exists():
            helpful_acts = torch.load(helpful_path)
            print(f"  helpful.pt: shape={helpful_acts.shape}, dtype={helpful_acts.dtype}")
            print(f"    Mean: {helpful_acts.mean():.4f}, Std: {helpful_acts.std():.4f}")
        else:
            print("  helpful.pt: NOT FOUND")
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

