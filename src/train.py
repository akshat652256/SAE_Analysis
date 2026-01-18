"""
Train Sparse Autoencoders on GPT-2 activations.

Usage:
    python train.py
    python train.py --model gpt2-medium --batch_size 4 --k 128
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAE on model activations")
    parser.add_argument("--model", type=str, default="gpt2", 
                       help="Model name (default: gpt2)")
    parser.add_argument("--dataset", type=str, default="EleutherAI/SmolLM2-135M-10B",
                       help="Dataset name (default: EleutherAI/SmolLM2-135M-10B)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size (default: 8)")
    parser.add_argument("--k", type=int, default=64,
                       help="Number of active features (default: 64)")
    parser.add_argument("--expansion_factor", type=int, default=32,
                       help="SAE expansion factor (default: 32)")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate (default: 3e-4)")
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Number of dataset samples (default: 10000)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/unnamed",
                       help="Checkpoint directory (default: checkpoints/unnamed)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Disable wandb
    os.environ["WANDB_MODE"] = "disabled"
    
    print("="*70)
    print("SPARSE AUTOENCODER TRAINING")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch Size: {args.batch_size}")
    print(f"K (sparsity): {args.k}")
    print(f"Expansion Factor: {args.expansion_factor}")
    print("="*70)
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": "cuda"},
        torch_dtype=torch.bfloat16,
    )
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=f"train[:{args.num_samples}]")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized = chunk_and_tokenize(dataset, tokenizer)
    
    # Configure SAE
    print("\nConfiguring SAE training...")
    sae_config = SaeConfig(
        k=args.k,
        expansion_factor=args.expansion_factor,
    )
    
    train_config = TrainConfig(
        sae_config,
        batch_size=args.batch_size,
        grad_acc_steps=1,
        lr=args.lr,
    )
    
    # Train
    print("\nStarting training...")
    print(f"Checkpoint will be saved to: {args.checkpoint_dir}")
    trainer = Trainer(train_config, tokenized, model)
    trainer.fit()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Checkpoints saved in: {args.checkpoint_dir}")
    print("="*70)


if __name__ == "__main__":
    main()