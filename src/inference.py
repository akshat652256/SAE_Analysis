"""
Run inference with trained SAE.

Usage:
    python inference.py
    python inference.py --checkpoint checkpoints/unnamed --layers 0 5 10
    python inference.py --text "Custom text to analyze"
"""

import argparse
import torch
from pathlib import Path
import json
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae, SaeConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAE inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/unnamed",
                       help="Checkpoint directory (default: checkpoints/unnamed)")
    parser.add_argument("--model", type=str, default="gpt2",
                       help="Model name (default: gpt2)")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5, 10],
                       help="Layers to analyze (default: 0 5 10)")
    parser.add_argument("--text", type=str, nargs="+", default=None,
                       help="Custom text to analyze (default: use example texts)")
    return parser.parse_args()


def load_sae(checkpoint_path, layer_name):
    """Load SAE from checkpoint directory."""
    layer_path = checkpoint_path / layer_name
    
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
    sae = sae.to("cuda").eval()
    
    return sae, d_in, sae_config


def analyze_text_with_sae(text, model, tokenizer, sae, d_in, sae_config, layer_idx):
    """Analyze a single text with SAE."""
    with torch.inference_mode():
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        outputs = model(**inputs, output_hidden_states=True)
        
        # Get activations for this layer
        hidden_state = outputs.hidden_states[layer_idx + 1]  # +1 for embedding
        hidden_state = hidden_state.flatten(0, 1)
        
        # Encode with SAE
        latents = sae.encode(hidden_state)
        
        # Reconstruct
        reconstructed = sae(hidden_state).sae_out
        
        # Calculate metrics
        num_active = latents.top_acts.shape[-1]
        total_features = sae_config.expansion_factor * d_in
        sparsity = 1.0 - (num_active / total_features)
        reconstruction_error = (hidden_state - reconstructed).pow(2).mean()
        
        # Top active features
        max_activations, _ = latents.top_acts.max(dim=0)
        top_feature_indices = latents.top_indices[0, max_activations.argsort(descending=True)[:5]]
        
        return {
            'sparsity': sparsity,
            'num_active': num_active,
            'total_features': total_features,
            'reconstruction_error': reconstruction_error.item(),
            'top_features': top_feature_indices.tolist()
        }


def main():
    args = parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    
    print("="*70)
    print("SPARSE AUTOENCODER INFERENCE")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model}")
    print(f"Layers to analyze: {args.layers}")
    print("="*70)
    
    # Load model and tokenizer
    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": "cuda"},
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Default test texts
    if args.text is None:
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Hello, world! This is a test."
        ]
    else:
        test_texts = [' '.join(args.text)]
    
    # Test first layer in detail
    first_layer = args.layers[0]
    print(f"\n{'='*70}")
    print(f"DETAILED ANALYSIS - Layer {first_layer}")
    print(f"{'='*70}")
    
    sae, d_in, sae_config = load_sae(checkpoint_path, f"h.{first_layer}")
    
    for text in test_texts:
        print(f"\nText: '{text}'")
        results = analyze_text_with_sae(text, model, tokenizer, sae, d_in, sae_config, first_layer)
        
        print(f"  Sparsity: {results['sparsity']:.2%}")
        print(f"  Active features: {results['num_active']} / {results['total_features']}")
        print(f"  Reconstruction error: {results['reconstruction_error']:.6f}")
        print(f"  Top 5 features: {results['top_features']}")
    
    # Compare across layers
    print(f"\n{'='*70}")
    print("CROSS-LAYER COMPARISON")
    print(f"{'='*70}")
    
    test_text = "The capital of France is Paris."
    inputs = tokenizer(test_text, return_tensors="pt").to("cuda")
    
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
        
        print(f"\nText: '{test_text}'")
        print("-" * 70)
        
        for layer_idx in args.layers:
            try:
                layer_path = checkpoint_path / f"h.{layer_idx}"
                if layer_path.exists():
                    sae_layer, layer_d_in, layer_config = load_sae(checkpoint_path, f"h.{layer_idx}")
                    
                    hidden_state = outputs.hidden_states[layer_idx + 1]
                    hidden_state = hidden_state.flatten(0, 1)
                    
                    latents = sae_layer.encode(hidden_state)
                    num_active = latents.top_acts.shape[-1]
                    
                    reconstructed = sae_layer(hidden_state).sae_out
                    reconstruction_error = (hidden_state - reconstructed).pow(2).mean()
                    
                    print(f"Layer {layer_idx}: {num_active} active features, "
                          f"Recon Error: {reconstruction_error:.6f}")
                else:
                    print(f"Layer {layer_idx}: Checkpoint not found")
            except Exception as e:
                print(f"Layer {layer_idx}: Error - {e}")
    
    print("\n" + "="*70)
    print("INFERENCE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()