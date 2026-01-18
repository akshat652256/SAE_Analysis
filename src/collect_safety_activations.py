"""
Collect model activations from safety datasets.

Usage:
    python collect_safety_activations.py
    python collect_safety_activations.py --model gpt2 --layers 6 7 8 9 10 --max_samples 1000
    python collect_safety_activations.py --datasets anthropic-hh --output_dir safety_activations
"""

import os
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Collect safety dataset activations")
    parser.add_argument("--model", type=str, default="gpt2",
                       help="Model name (default: gpt2)")
    parser.add_argument("--datasets", type=str, nargs="+", 
                       default=["anthropic-hh"],
                       choices=["anthropic-hh", "harmbench"],
                       help="Safety datasets to use (default: anthropic-hh)")
    parser.add_argument("--layers", type=int, nargs="+", 
                       default=[6, 7, 8, 9, 10],
                       help="Layers to extract activations from (default: 6 7 8 9 10)")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="Max samples per category (default: 1000)")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Max token length (default: 512)")
    parser.add_argument("--output_dir", type=str, default="safety_activations",
                       help="Output directory (default: safety_activations)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for processing (default: 8)")
    parser.add_argument("--num_examples", type=int, default=5,
                       help="Number of example texts to print (default: 5)")
    return parser.parse_args()


def print_examples(examples, label, num_examples=5):
    """Print sample examples from the dataset."""
    print(f"\n{'='*70}")
    print(f"SAMPLE {label.upper()} EXAMPLES")
    print(f"{'='*70}")
    
    for i, example in enumerate(examples[:num_examples]):
        # Truncate long examples for display
        display_text = example[:200] + "..." if len(example) > 200 else example
        print(f"\n[{label.upper()} Example {i+1}]")
        print(f"{display_text}")
        print("-" * 70)


def load_anthropic_hh(split="train", max_samples=None):
    """
    Load Anthropic HH dataset.
    Returns harmful and helpful examples separately.
    """
    print("\nLoading Anthropic Helpful-Harmless dataset...")
    
    try:
        # Load the dataset
        dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        
        harmful_examples = []
        helpful_examples = []
        
        for idx, example in enumerate(dataset):
            if max_samples and len(harmful_examples) >= max_samples and len(helpful_examples) >= max_samples:
                break
            
            # The dataset has 'chosen' (helpful) and 'rejected' (harmful) responses
            # Extract the response part after "Assistant:"
            
            chosen_text = example.get('chosen', '')
            rejected_text = example.get('rejected', '')
            
            # Add helpful example (chosen response)
            if chosen_text and len(helpful_examples) < max_samples:
                # Extract just the assistant's response
                if "Assistant:" in chosen_text:
                    response = chosen_text.split("Assistant:")[-1].strip()
                    helpful_examples.append(response)
            
            # Add harmful example (rejected response)
            if rejected_text and len(harmful_examples) < max_samples:
                if "Assistant:" in rejected_text:
                    response = rejected_text.split("Assistant:")[-1].strip()
                    harmful_examples.append(response)
        
        print(f"Loaded {len(harmful_examples)} harmful examples")
        print(f"Loaded {len(helpful_examples)} helpful examples")
        
        return harmful_examples, helpful_examples
    
    except Exception as e:
        print(f"Error loading Anthropic HH: {e}")
        print("Using dummy data for testing...")
        
        # Fallback dummy data for testing
        harmful_examples = [
            "I can help you hack into someone's computer system.",
            "Here's how to make a dangerous weapon at home.",
            "I'll teach you how to manipulate people for personal gain.",
            "Let me show you how to bypass security systems.",
            "I can provide instructions for harmful activities.",
        ] * (max_samples // 5)
        
        helpful_examples = [
            "I'd be happy to help you learn programming in a safe and ethical way.",
            "Here are some healthy recipes you might enjoy.",
            "I can provide information about mental health resources.",
            "Let me explain computer security best practices.",
            "I can help you with educational information.",
        ] * (max_samples // 5)
        
        return harmful_examples[:max_samples], helpful_examples[:max_samples]


def load_harmbench(max_samples=None):
    """
    Load HarmBench dataset.
    Returns harmful prompts/completions.
    """
    print("\nLoading HarmBench dataset...")
    
    try:
        # HarmBench contains adversarial harmful prompts
        dataset = load_dataset("harmbench/harmbench_text", split="test")
        
        harmful_examples = []
        
        for idx, example in enumerate(dataset):
            if max_samples and len(harmful_examples) >= max_samples:
                break
            
            # Get the behavior or prompt
            text = example.get('behavior', '') or example.get('prompt', '')
            if text:
                harmful_examples.append(text)
        
        print(f"Loaded {len(harmful_examples)} harmful examples from HarmBench")
        
        return harmful_examples, []
    
    except Exception as e:
        print(f"Error loading HarmBench: {e}")
        print("Skipping HarmBench...")
        return [], []


def extract_activations_batch(model, tokenizer, texts, layers, max_length=512):
    """
    Extract activations for a batch of texts at specified layers.
    
    Returns:
        dict: {layer_idx: activations_tensor}
    """
    # Tokenize batch
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(model.device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Extract activations for each layer
    activations = {}
    for layer_idx in layers:
        # hidden_states[0] is embedding, hidden_states[1] is layer 0, etc.
        hidden_state = outputs.hidden_states[layer_idx + 1]
        
        # Average pool over sequence length: (batch, seq_len, hidden_dim) -> (batch, hidden_dim)
        # This gives us one activation vector per example
        pooled = hidden_state.mean(dim=1)
        
        activations[layer_idx] = pooled.cpu()
    
    return activations


def collect_and_save_activations(
    model, 
    tokenizer, 
    examples, 
    label, 
    layers, 
    output_dir, 
    batch_size=8,
    max_length=512
):
    """
    Collect activations for all examples and save by layer.
    
    Args:
        examples: List of text examples
        label: "harmful" or "helpful"
        layers: List of layer indices
        output_dir: Where to save activations
    """
    print(f"\nCollecting activations for {len(examples)} {label} examples...")
    
    # Initialize storage for each layer
    layer_activations = {layer_idx: [] for layer_idx in layers}
    
    # Process in batches
    for i in tqdm(range(0, len(examples), batch_size), desc=f"Processing {label}"):
        batch = examples[i:i + batch_size]
        
        # Extract activations for this batch
        batch_activations = extract_activations_batch(
            model, tokenizer, batch, layers, max_length
        )
        
        # Accumulate activations for each layer
        for layer_idx in layers:
            layer_activations[layer_idx].append(batch_activations[layer_idx])
    
    # Concatenate and save for each layer
    for layer_idx in layers:
        # Concatenate all batches: list of (batch, hidden_dim) -> (total_examples, hidden_dim)
        all_activations = torch.cat(layer_activations[layer_idx], dim=0)
        
        # Create directory for this layer
        layer_dir = Path(output_dir) / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        # Save activations
        save_path = layer_dir / f"{label}.pt"
        torch.save(all_activations, save_path)
        
        print(f"  Layer {layer_idx}: Saved {all_activations.shape} to {save_path}")
    
    return layer_activations


def save_metadata(output_dir, args, harmful_count, helpful_count):
    """Save metadata about the collection process."""
    metadata = {
        "model": args.model,
        "datasets": args.datasets,
        "layers": args.layers,
        "max_samples": args.max_samples,
        "max_length": args.max_length,
        "harmful_examples": harmful_count,
        "helpful_examples": helpful_count,
    }
    
    metadata_path = Path(output_dir) / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to {metadata_path}")


def save_example_texts(output_dir, harmful_examples, helpful_examples, num_examples=10):
    """Save example texts to a file for reference."""
    examples_path = Path(output_dir) / "example_texts.txt"
    
    with open(examples_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("HARMFUL EXAMPLES\n")
        f.write("="*70 + "\n\n")
        
        for i, example in enumerate(harmful_examples[:num_examples]):
            f.write(f"[Harmful Example {i+1}]\n")
            f.write(f"{example}\n")
            f.write("-"*70 + "\n\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("HELPFUL EXAMPLES\n")
        f.write("="*70 + "\n\n")
        
        for i, example in enumerate(helpful_examples[:num_examples]):
            f.write(f"[Helpful Example {i+1}]\n")
            f.write(f"{example}\n")
            f.write("-"*70 + "\n\n")
    
    print(f"Example texts saved to {examples_path}")


def main():
    args = parse_args()
    
    print("="*70)
    print("SAFETY ACTIVATION COLLECTION")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Datasets: {args.datasets}")
    print(f"Layers: {args.layers}")
    print(f"Max samples: {args.max_samples}")
    print(f"Output directory: {args.output_dir}")
    print("="*70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": "cuda" if torch.cuda.is_available() else "cpu"},
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    all_harmful = []
    all_helpful = []
    
    for dataset_name in args.datasets:
        if dataset_name == "anthropic-hh":
            harmful, helpful = load_anthropic_hh(max_samples=args.max_samples)
            all_harmful.extend(harmful)
            all_helpful.extend(helpful)
        elif dataset_name == "harmbench":
            harmful, _ = load_harmbench(max_samples=args.max_samples)
            all_harmful.extend(harmful)
    
    # Limit to max_samples
    all_harmful = all_harmful[:args.max_samples]
    all_helpful = all_helpful[:args.max_samples]
    
    print(f"\nTotal harmful examples: {len(all_harmful)}")
    print(f"Total helpful examples: {len(all_helpful)}")
    
    # Print sample examples
    if all_harmful:
        print_examples(all_harmful, "harmful", args.num_examples)
    
    if all_helpful:
        print_examples(all_helpful, "helpful", args.num_examples)
    
    # Save example texts to file
    # save_example_texts(args.output_dir, all_harmful, all_helpful, num_examples=20)
    
    # Collect activations for harmful examples
    if all_harmful:
        collect_and_save_activations(
            model, tokenizer, all_harmful, "harmful", 
            args.layers, args.output_dir, args.batch_size, args.max_length
        )
    
    # Collect activations for helpful examples
    if all_helpful:
        collect_and_save_activations(
            model, tokenizer, all_helpful, "helpful",
            args.layers, args.output_dir, args.batch_size, args.max_length
        )
    
    # Save metadata
    save_metadata(args.output_dir, args, len(all_harmful), len(all_helpful))
    
    print("\n" + "="*70)
    print("COLLECTION COMPLETE!")
    print(f"Activations saved to: {args.output_dir}")
    print("="*70)
    
    # Print summary
    print("\nSummary:")
    for layer_idx in args.layers:
        print(f"  layer_{layer_idx}/")
        if all_harmful:
            print(f"    - harmful.pt ({len(all_harmful)} examples)")
        if all_helpful:
            print(f"    - helpful.pt ({len(all_helpful)} examples)")
    
    print(f"\n  example_texts.txt (20 examples from each category)")
    print(f"  metadata.json")


if __name__ == "__main__":
    main()