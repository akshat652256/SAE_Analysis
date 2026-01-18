"""
Compute feature importance ratios for safety analysis.

Usage:
    python compute_feature_importance.py
    python compute_feature_importance.py --encoded_dir encoded_features --output_dir importance_analysis
    python compute_feature_importance.py --layers 6 7 8 9 10 --top_k 100
"""

import argparse
import torch
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Compute feature importance for safety")
    parser.add_argument("--encoded_dir", type=str, default="encoded_features",
                       help="Encoded features directory (default: encoded_features)")
    parser.add_argument("--output_dir", type=str, default="importance_analysis",
                       help="Output directory (default: importance_analysis)")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                       help="Layers to analyze (default: all layers found)")
    parser.add_argument("--method", type=str, default="ratio",
                       choices=["ratio", "difference", "log_ratio"],
                       help="Importance computation method (default: ratio)")
    parser.add_argument("--top_k", type=int, default=100,
                       help="Number of top features to save (default: 100)")
    return parser.parse_args()


def compute_activation_rates(top_indices, num_features):
    """
    Compute activation rate for each feature.
    
    Args:
        top_indices: (num_examples, k) tensor of active feature indices
        num_features: Total number of features in SAE
    
    Returns:
        activation_rates: (num_features,) tensor with activation rate for each feature
        activation_counts: (num_features,) tensor with raw counts
    """
    num_examples = top_indices.shape[0]
    
    # Count how many times each feature appears
    activation_counts = torch.zeros(num_features, dtype=torch.long)
    
    # Flatten indices and count occurrences
    for feature_idx in top_indices.flatten():
        activation_counts[feature_idx] += 1
    
    # Compute activation rate (proportion of examples where feature is active)
    activation_rates = activation_counts.float() / num_examples
    
    return activation_rates, activation_counts


def compute_mean_activation_strength(top_acts, top_indices, num_features):
    """
    Compute mean activation strength for each feature when it's active.
    
    Args:
        top_acts: (num_examples, k) tensor of activation values
        top_indices: (num_examples, k) tensor of feature indices
        num_features: Total number of features
    
    Returns:
        mean_strengths: (num_features,) tensor with mean activation strength
    """
    # Initialize accumulators
    activation_sums = torch.zeros(num_features, dtype=torch.float32)
    activation_counts = torch.zeros(num_features, dtype=torch.long)
    
    # Accumulate activation values for each feature
    num_examples, k = top_acts.shape
    
    for i in range(num_examples):
        for j in range(k):
            feature_idx = top_indices[i, j].item()
            activation_val = top_acts[i, j].item()
            
            activation_sums[feature_idx] += activation_val
            activation_counts[feature_idx] += 1
    
    # Compute mean (avoiding division by zero)
    mean_strengths = torch.zeros(num_features, dtype=torch.float32)
    mask = activation_counts > 0
    mean_strengths[mask] = activation_sums[mask] / activation_counts[mask].float()
    
    return mean_strengths


def compute_importance_scores(
    harmful_rates, 
    helpful_rates, 
    harmful_strengths,
    helpful_strengths,
    method="ratio"
):
    """
    Compute importance scores for features.
    
    Args:
        harmful_rates: Activation rates on harmful examples
        helpful_rates: Activation rates on helpful examples
        harmful_strengths: Mean activation strengths on harmful examples
        helpful_strengths: Mean activation strengths on helpful examples
        method: "ratio", "difference", or "log_ratio"
    
    Returns:
        importance_scores: Tensor of importance scores
    """
    epsilon = 1e-8  # For numerical stability
    
    if method == "ratio":
        # Simple ratio: harmful_rate / helpful_rate
        # Higher = more important for harmful content
        importance = harmful_rates / (helpful_rates + epsilon)
    
    elif method == "difference":
        # Difference: harmful_rate - helpful_rate
        importance = harmful_rates - helpful_rates
    
    elif method == "log_ratio":
        # Log ratio: log(harmful_rate / helpful_rate)
        importance = torch.log((harmful_rates + epsilon) / (helpful_rates + epsilon))
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return importance


def analyze_layer(encoded_dir, layer_idx, method="ratio"):
    """
    Analyze features for a single layer.
    
    Returns:
        DataFrame with feature statistics
    """
    layer_dir = Path(encoded_dir) / f"layer_{layer_idx}"
    
    # Load encoded features
    harmful_path = layer_dir / "harmful_encoded.pt"
    helpful_path = layer_dir / "helpful_encoded.pt"
    
    if not harmful_path.exists() or not helpful_path.exists():
        print(f"Warning: Encoded features not found for layer {layer_idx}")
        return None
    
    harmful_data = torch.load(harmful_path)
    helpful_data = torch.load(helpful_path)
    
    harmful_acts = harmful_data['top_acts']
    harmful_indices = harmful_data['top_indices']
    helpful_acts = helpful_data['top_acts']
    helpful_indices = helpful_data['top_indices']
    
    # Infer number of features from indices
    num_features = max(harmful_indices.max().item(), helpful_indices.max().item()) + 1
    
    print(f"\nAnalyzing layer {layer_idx}:")
    print(f"  Harmful examples: {harmful_data['num_examples']}")
    print(f"  Helpful examples: {helpful_data['num_examples']}")
    print(f"  Total features: {num_features}")
    print(f"  k (active per example): {harmful_acts.shape[1]}")
    
    # Compute activation rates
    print("  Computing activation rates...")
    harmful_rates, harmful_counts = compute_activation_rates(harmful_indices, num_features)
    helpful_rates, helpful_counts = compute_activation_rates(helpful_indices, num_features)
    
    # Compute mean activation strengths
    print("  Computing activation strengths...")
    harmful_strengths = compute_mean_activation_strength(harmful_acts, harmful_indices, num_features)
    helpful_strengths = compute_mean_activation_strength(helpful_acts, helpful_indices, num_features)
    
    # Compute importance scores
    print(f"  Computing importance scores (method: {method})...")
    importance_scores = compute_importance_scores(
        harmful_rates, helpful_rates,
        harmful_strengths, helpful_strengths,
        method=method
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature_idx': range(num_features),
        'harmful_activation_rate': harmful_rates.numpy(),
        'helpful_activation_rate': helpful_rates.numpy(),
        'harmful_activation_count': harmful_counts.numpy(),
        'helpful_activation_count': helpful_counts.numpy(),
        'harmful_mean_strength': harmful_strengths.numpy(),
        'helpful_mean_strength': helpful_strengths.numpy(),
        'importance_score': importance_scores.numpy(),
    })
    
    # Add derived metrics
    df['rate_difference'] = df['harmful_activation_rate'] - df['helpful_activation_rate']
    df['total_activations'] = df['harmful_activation_count'] + df['helpful_activation_count']
    
    # Sort by importance score
    df = df.sort_values('importance_score', ascending=False).reset_index(drop=True)
    
    return df


def save_analysis(df, layer_idx, output_dir, top_k=100):
    """Save analysis results."""
    layer_dir = Path(output_dir) / f"layer_{layer_idx}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    full_path = layer_dir / "feature_importance_full.csv"
    df.to_csv(full_path, index=False)
    print(f"  Saved full analysis to {full_path}")
    
    # Save top-k most important features
    top_df = df.head(top_k)
    top_path = layer_dir / f"feature_importance_top{top_k}.csv"
    top_df.to_csv(top_path, index=False)
    print(f"  Saved top {top_k} features to {top_path}")
    
    # Save summary statistics
    summary = {
        'layer': layer_idx,
        'total_features': len(df),
        'features_active_on_harmful': int((df['harmful_activation_count'] > 0).sum()),
        'features_active_on_helpful': int((df['helpful_activation_count'] > 0).sum()),
        'top_10_features': top_df.head(10)['feature_idx'].tolist(),
        'top_10_importance_scores': top_df.head(10)['importance_score'].tolist(),
    }
    
    summary_path = layer_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to {summary_path}")
    
    return top_df


def print_top_features(df, layer_idx, top_n=10):
    """Print top safety features."""
    print(f"\n{'='*70}")
    print(f"TOP {top_n} SAFETY FEATURES - LAYER {layer_idx}")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Feature':<10} {'Importance':<12} {'Harmful Rate':<14} {'Helpful Rate':<14}")
    print("-" * 70)
    
    for i, row in df.head(top_n).iterrows():
        print(f"{i+1:<6} {row['feature_idx']:<10} "
              f"{row['importance_score']:<12.4f} "
              f"{row['harmful_activation_rate']:<14.4f} "
              f"{row['helpful_activation_rate']:<14.4f}")


def get_available_layers(encoded_dir):
    """Get list of available layers."""
    encoded_path = Path(encoded_dir)
    layer_dirs = [d for d in encoded_path.iterdir() 
                  if d.is_dir() and d.name.startswith("layer_")]
    
    layers = []
    for layer_dir in sorted(layer_dirs):
        layer_num = int(layer_dir.name.split("_")[1])
        layers.append(layer_num)
    
    return layers


def save_global_summary(output_dir, layer_results, method):
    """Save summary across all layers."""
    summary = {
        'method': method,
        'layers_analyzed': sorted(layer_results.keys()),
        'layer_summaries': {}
    }
    
    for layer_idx in sorted(layer_results.keys()):
        df = layer_results[layer_idx]
        top_10 = df.head(10)
        
        summary['layer_summaries'][f'layer_{layer_idx}'] = {
            'total_features': len(df),
            'top_10_features': top_10['feature_idx'].tolist(),
            'top_10_scores': top_10['importance_score'].tolist(),
            'max_importance': float(df['importance_score'].max()),
            'min_importance': float(df['importance_score'].min()),
        }
    
    summary_path = Path(output_dir) / "global_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nGlobal summary saved to {summary_path}")


def main():
    args = parse_args()
    
    print("="*70)
    print("COMPUTE FEATURE IMPORTANCE FOR SAFETY")
    print("="*70)
    print(f"Encoded features directory: {args.encoded_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Method: {args.method}")
    print(f"Top-k to save: {args.top_k}")
    print("="*70)
    
    encoded_dir = Path(args.encoded_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which layers to process
    if args.layers is None:
        layers = get_available_layers(encoded_dir)
        print(f"\nAuto-detected layers: {layers}")
    else:
        layers = args.layers
        print(f"\nProcessing specified layers: {layers}")
    
    if not layers:
        print("Error: No layers found to process!")
        return
    
    # Analyze each layer
    layer_results = {}
    
    for layer_idx in layers:
        print(f"\n{'='*70}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*70}")
        
        df = analyze_layer(encoded_dir, layer_idx, method=args.method)
        
        if df is not None:
            # Save results
            save_analysis(df, layer_idx, output_dir, top_k=args.top_k)
            
            # Print top features
            print_top_features(df, layer_idx, top_n=10)
            
            layer_results[layer_idx] = df
    
    # Save global summary
    if layer_results:
        save_global_summary(output_dir, layer_results, args.method)
    
    print("\n" + "="*70)
    print("IMPORTANCE ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("="*70)
    
    # Print final summary
    print("\nOutput structure:")
    for layer_idx in sorted(layer_results.keys()):
        print(f"  layer_{layer_idx}/")
        print(f"    - feature_importance_full.csv")
        print(f"    - feature_importance_top{args.top_k}.csv")
        print(f"    - summary.json")
    print(f"  global_summary.json")


if __name__ == "__main__":
    main()