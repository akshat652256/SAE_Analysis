"""
Encode safety activations with trained SAE to get sparse feature vectors.

Usage:
    python encode_safety_activations.py
    python encode_safety_activations.py --sae_checkpoint checkpoints/unnamed --activations_dir safety_activations
    python encode_safety_activations.py --layers 6 7 8 9 10 --output_dir encoded_features
"""

import argparse
import torch
from pathlib import Path
import json
from tqdm import tqdm
from safetensors.torch import load_file
from sparsify import Sae, SaeConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Encode safety activations with SAE")
    parser.add_argument("--sae_checkpoint", type=str, default="checkpoints/unnamed",
                       help="SAE checkpoint directory (default: checkpoints/unnamed)")
    parser.add_argument("--activations_dir", type=str, default="safety_activations",
                       help="Safety activations directory (default: safety_activations)")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                       help="Layers to process (default: all layers found)")
    parser.add_argument("--output_dir", type=str, default="encoded_features",
                       help="Output directory for encoded features (default: encoded_features)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for encoding (default: 32)")
    return parser.parse_args()


def load_sae(checkpoint_path, layer_name):
    """Load SAE from checkpoint directory."""
    layer_path = checkpoint_path / layer_name
    
    if not layer_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {layer_path}")
    
    # Load config
    with open(layer_path / "cfg.json", "r") as f:
        config = json.load(f)
    
    # Load weights
    state_dict = load_file(layer_path / "sae.safetensors")
    
    # Create SAE
    d_in = config.pop('d_in')
    sae_config = SaeConfig(**config)
    sae = Sae(d_in, sae_config)
    sae.load_state_dict(state_dict)
    sae = sae.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    
    return sae, d_in, sae_config


def encode_activations_batch(sae, activations_batch):
    """
    Encode a batch of activations with SAE.
    
    Args:
        sae: Trained SAE model
        activations_batch: (batch_size, hidden_dim) tensor
    
    Returns:
        dict with top_acts, top_indices for the batch
    """
    device = next(sae.parameters()).device
    activations_batch = activations_batch.to(device)
    
    with torch.no_grad():
        # Encode to get sparse features
        encoder_output = sae.encode(activations_batch)
        
        # encoder_output has: top_acts, top_indices, pre_acts
        # top_acts: (batch, k) - values of top-k active features
        # top_indices: (batch, k) - indices of top-k active features
        
        return {
            'top_acts': encoder_output.top_acts.cpu(),
            'top_indices': encoder_output.top_indices.cpu(),
        }


def encode_and_save(sae, activations, label, layer_idx, output_dir, batch_size=32):
    """
    Encode all activations and save sparse feature vectors.
    
    Args:
        sae: Trained SAE
        activations: (num_examples, hidden_dim) tensor
        label: "harmful" or "helpful"
        layer_idx: Layer index
        output_dir: Where to save
        batch_size: Batch size for encoding
    
    Returns:
        dict with aggregated top_acts and top_indices
    """
    print(f"\nEncoding {len(activations)} {label} examples for layer {layer_idx}...")
    
    all_top_acts = []
    all_top_indices = []
    
    # Process in batches
    num_batches = (len(activations) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(activations), batch_size), 
                  desc=f"Encoding {label}", total=num_batches):
        batch = activations[i:i + batch_size]
        
        # Encode batch
        encoded = encode_activations_batch(sae, batch)
        
        all_top_acts.append(encoded['top_acts'])
        all_top_indices.append(encoded['top_indices'])
    
    # Concatenate all batches
    top_acts = torch.cat(all_top_acts, dim=0)
    top_indices = torch.cat(all_top_indices, dim=0)
    
    # Create output directory
    layer_dir = Path(output_dir) / f"layer_{layer_idx}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    
    # Save encoded features
    encoded_features = {
        'top_acts': top_acts,
        'top_indices': top_indices,
        'num_examples': len(activations),
    }
    
    save_path = layer_dir / f"{label}_encoded.pt"
    torch.save(encoded_features, save_path)
    
    print(f"  Saved to {save_path}")
    print(f"  Shape: top_acts={top_acts.shape}, top_indices={top_indices.shape}")
    
    return encoded_features


def get_available_layers(activations_dir):
    """Get list of available layers from activations directory."""
    activations_path = Path(activations_dir)
    layer_dirs = [d for d in activations_path.iterdir() 
                  if d.is_dir() and d.name.startswith("layer_")]
    
    layers = []
    for layer_dir in sorted(layer_dirs):
        layer_num = int(layer_dir.name.split("_")[1])
        layers.append(layer_num)
    
    return layers


def save_encoding_metadata(output_dir, sae_checkpoint, activations_dir, layers, k):
    """Save metadata about the encoding process."""
    metadata = {
        "sae_checkpoint": str(sae_checkpoint),
        "activations_dir": str(activations_dir),
        "layers": layers,
        "k": k,
        "description": "Sparse feature vectors from encoding safety activations with trained SAE"
    }
    
    metadata_path = Path(output_dir) / "encoding_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to {metadata_path}")


def main():
    args = parse_args()
    
    print("="*70)
    print("ENCODE SAFETY ACTIVATIONS WITH SAE")
    print("="*70)
    print(f"SAE checkpoint: {args.sae_checkpoint}")
    print(f"Activations directory: {args.activations_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*70)
    
    sae_checkpoint = Path(args.sae_checkpoint)
    activations_dir = Path(args.activations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which layers to process
    if args.layers is None:
        layers = get_available_layers(activations_dir)
        print(f"\nAuto-detected layers: {layers}")
    else:
        layers = args.layers
        print(f"\nProcessing specified layers: {layers}")
    
    if not layers:
        print("Error: No layers found to process!")
        return
    
    # Track K value (will be same for all layers)
    k_value = None
    
    # Process each layer
    for layer_idx in layers:
        print(f"\n{'='*70}")
        print(f"PROCESSING LAYER {layer_idx}")
        print(f"{'='*70}")
        
        # Load SAE for this layer
        try:
            sae, d_in, sae_config = load_sae(sae_checkpoint, f"h.{layer_idx}")
            print(f"Loaded SAE for layer {layer_idx}")
            print(f"  d_in: {d_in}")
            print(f"  k: {sae_config.k}")
            print(f"  expansion_factor: {sae_config.expansion_factor}")
            print(f"  total_features: {d_in * sae_config.expansion_factor}")
            
            if k_value is None:
                k_value = sae_config.k
        
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print(f"Skipping layer {layer_idx}")
            continue
        
        # Load activations for this layer
        layer_act_dir = activations_dir / f"layer_{layer_idx}"
        
        # Process harmful activations
        harmful_path = layer_act_dir / "harmful.pt"
        if harmful_path.exists():
            harmful_acts = torch.load(harmful_path)
            print(f"\nLoaded harmful activations: {harmful_acts.shape}")
            
            encode_and_save(
                sae, harmful_acts, "harmful", layer_idx, 
                output_dir, args.batch_size
            )
        else:
            print(f"\nWarning: Harmful activations not found at {harmful_path}")
        
        # Process helpful activations
        helpful_path = layer_act_dir / "helpful.pt"
        if helpful_path.exists():
            helpful_acts = torch.load(helpful_path)
            print(f"\nLoaded helpful activations: {helpful_acts.shape}")
            
            encode_and_save(
                sae, helpful_acts, "helpful", layer_idx,
                output_dir, args.batch_size
            )
        else:
            print(f"\nWarning: Helpful activations not found at {helpful_path}")
    
    # Save metadata
    save_encoding_metadata(output_dir, sae_checkpoint, activations_dir, layers, k_value)
    
    print("\n" + "="*70)
    print("ENCODING COMPLETE!")
    print(f"Encoded features saved to: {output_dir}")
    print("="*70)
    
    # Print summary
    print("\nOutput structure:")
    for layer_idx in layers:
        layer_dir = output_dir / f"layer_{layer_idx}"
        if layer_dir.exists():
            print(f"  layer_{layer_idx}/")
            if (layer_dir / "harmful_encoded.pt").exists():
                print(f"    - harmful_encoded.pt")
            if (layer_dir / "helpful_encoded.pt").exists():
                print(f"    - helpful_encoded.pt")


if __name__ == "__main__":
    main()