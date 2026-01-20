"""
Diagnose encoded features to understand activation patterns.

Usage:
    python diagnose_encoded_features.py
    python diagnose_encoded_features.py --encoded_dir encoded_features --layer 10
"""

import argparse
import torch
from pathlib import Path
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose encoded features")
    parser.add_argument("--encoded_dir", type=str, default="encoded_features",
                       help="Encoded features directory (default: encoded_features)")
    parser.add_argument("--layer", type=int, default=10,
                       help="Layer to diagnose (default: 10)")
    return parser.parse_args()


def analyze_encoded_data(data, label):
    """Analyze encoded feature data."""
    print(f"\n{'='*70}")
    print(f"{label.upper()} EXAMPLES ANALYSIS")
    print(f"{'='*70}")
    
    top_acts = data['top_acts']
    top_indices = data['top_indices']
    
    print(f"Number of examples: {data['num_examples']}")
    print(f"Top acts shape: {top_acts.shape}")
    print(f"Top indices shape: {top_indices.shape}")
    
    print(f"\nTop acts statistics:")
    print(f"  Min: {top_acts.min():.6f}")
    print(f"  Max: {top_acts.max():.6f}")
    print(f"  Mean: {top_acts.mean():.6f}")
    print(f"  Std: {top_acts.std():.6f}")
    
    print(f"\nTop indices statistics:")
    print(f"  Min index: {top_indices.min()}")
    print(f"  Max index: {top_indices.max()}")
    print(f"  Unique features used: {len(torch.unique(top_indices))}")
    
    # Check if there are any zero activations
    zero_acts = (top_acts == 0).sum()
    print(f"\nZero activations: {zero_acts} / {top_acts.numel()} ({100*zero_acts/top_acts.numel():.2f}%)")
    
    # Show sample from first few examples
    print(f"\nSample from first 3 examples:")
    for i in range(min(3, len(top_acts))):
        print(f"\n  Example {i}:")
        print(f"    Indices: {top_indices[i][:10].tolist()}...")
        print(f"    Values:  {top_acts[i][:10].tolist()}...")
    
    # Check for overlap in feature usage
    all_features = top_indices.flatten()
    unique_features = torch.unique(all_features)
    feature_counts = torch.bincount(all_features.long(), minlength=top_indices.max()+1)
    
    print(f"\nFeature usage distribution:")
    print(f"  Total unique features activated: {len(unique_features)}")
    print(f"  Most activated feature: {feature_counts.argmax()} (count: {feature_counts.max()})")
    print(f"  Least activated feature: {feature_counts[feature_counts > 0].argmin()} (count: {feature_counts[feature_counts > 0].min()})")
    
    # Top 10 most frequently activated features
    top_10_counts, top_10_features = torch.topk(feature_counts, 10)
    print(f"\n  Top 10 most frequently activated features:")
    for feat, count in zip(top_10_features.tolist(), top_10_counts.tolist()):
        print(f"    Feature {feat}: activated {count} times")
    
    return unique_features, feature_counts


def compare_feature_overlap(harmful_features, helpful_features, harmful_counts, helpful_counts):
    """Compare feature usage between harmful and helpful."""
    print(f"\n{'='*70}")
    print("FEATURE OVERLAP ANALYSIS")
    print(f"{'='*70}")
    
    harmful_set = set(harmful_features.tolist())
    helpful_set = set(helpful_features.tolist())
    
    overlap = harmful_set & helpful_set
    only_harmful = harmful_set - helpful_set
    only_helpful = helpful_set - harmful_set
    
    print(f"Features only in harmful: {len(only_harmful)}")
    print(f"Features only in helpful: {len(only_helpful)}")
    print(f"Features in both: {len(overlap)}")
    
    print(f"\nSample features only in harmful (first 20):")
    print(f"  {sorted(list(only_harmful))[:20]}")
    
    print(f"\nSample features only in helpful (first 20):")
    print(f"  {sorted(list(only_helpful))[:20]}")
    
    print(f"\nSample overlapping features (first 20):")
    print(f"  {sorted(list(overlap))[:20]}")
    
    # For overlapping features, compare activation frequencies
    if len(overlap) > 0:
        print(f"\nActivation frequency comparison for overlapping features:")
        overlap_list = sorted(list(overlap))[:10]
        
        print(f"{'Feature':<10} {'Harmful Count':<15} {'Helpful Count':<15} {'Ratio':<10}")
        print("-" * 60)
        
        for feat in overlap_list:
            h_count = harmful_counts[feat].item()
            he_count = helpful_counts[feat].item()
            ratio = h_count / max(he_count, 1)
            print(f"{feat:<10} {h_count:<15} {he_count:<15} {ratio:<10.2f}")


def main():
    args = parse_args()
    
    print("="*70)
    print("ENCODED FEATURES DIAGNOSTIC")
    print("="*70)
    print(f"Encoded directory: {args.encoded_dir}")
    print(f"Layer: {args.layer}")
    print("="*70)
    
    layer_dir = Path(args.encoded_dir) / f"layer_{args.layer}"
    
    # Load harmful data
    harmful_path = layer_dir / "harmful_encoded.pt"
    if not harmful_path.exists():
        print(f"Error: {harmful_path} not found!")
        return
    
    harmful_data = torch.load(harmful_path)
    harmful_features, harmful_counts = analyze_encoded_data(harmful_data, "harmful")
    
    # Load helpful data
    helpful_path = layer_dir / "helpful_encoded.pt"
    if not helpful_path.exists():
        print(f"Error: {helpful_path} not found!")
        return
    
    helpful_data = torch.load(helpful_path)
    helpful_features, helpful_counts = analyze_encoded_data(helpful_data, "helpful")
    
    # Compare
    compare_feature_overlap(harmful_features, helpful_features, harmful_counts, helpful_counts)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()