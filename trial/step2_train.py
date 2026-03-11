#!/usr/bin/env python3
"""
Emotion-GSRM: Step 2 — SFT Training
======================================
Fine-tunes Qwen2.5-Omni-7B on synthesized CoT data.

Usage:
    python step2_train.py --data_path ./meld_cot_output/sft_training_data.jsonl

Options:
    --num_gpus N        Number of GPUs (default: 1)
    --epochs N          Training epochs (default: 10)
    --lr RATE           Learning rate (default: 2e-5)
    --lora_rank N       LoRA rank (default: 64)
    --output_dir DIR    Checkpoint directory (default: ./emotion_gsrm_checkpoints)
    --dry_run           Only prepare data and print command, don't train

For a quick test with RAVDESS (~100 samples), training takes:
    - ~30 min on a single A100-80GB
    - ~15 min on 4x A100-40GB
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("step2")


def main():
    parser = argparse.ArgumentParser(description="SFT training for Emotion-GSRM")
    parser.add_argument("--data_path", type=str, 
                        default="/home/azureuser/atik/speechRL_backup/meld_cot_output/sft_training_data.jsonl",
                        help="Path to sft_training_data.jsonl from Step 1")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--output_dir", type=str,
                        default="/home/azureuser/atik/speechRL_backup/emotion_gsrm_checkpoints")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only prepare data and print command")
    parser.add_argument("--framework", type=str, default="swift",
                        choices=["swift", "huggingface"])
    args = parser.parse_args()

    from pathlib import Path
    if not Path(args.data_path).exists():
        print(f"ERROR: Data file not found: {args.data_path}")
        print("Run step1_synthesize.py first.")
        sys.exit(1)

    # Count samples
    n_samples = sum(1 for line in open(args.data_path) if line.strip())
    logger.info(f"Training data: {n_samples} samples from {args.data_path}")

    from sft.config import TrainingConfig, TrainingFramework
    from sft.trainer import EmotionGSRMTrainer
    from sft.dataset import SFTDataset

    # Configure
    config = TrainingConfig(
        train_data_path=args.data_path,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_gpus=args.num_gpus,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        output_dir=args.output_dir,
        framework=(
            TrainingFramework.SWIFT if args.framework == "swift"
            else TrainingFramework.HUGGINGFACE
        ),
        use_wandb=False,  # Disable for initial testing
    )

    effective_batch = args.batch_size * args.grad_accum * args.num_gpus
    logger.info(f"Effective batch size: {effective_batch}")
    logger.info(f"Steps per epoch: {n_samples // effective_batch}")
    logger.info(f"Total steps: {(n_samples // effective_batch) * args.epochs}")

    # Validate
    warnings = config.validate()
    for w in warnings:
        logger.warning(w)

    # Prepare trainer
    dataset = SFTDataset(
        data_path=args.data_path,
        val_split=0.1,
    ).load()

    trainer = EmotionGSRMTrainer(config, dataset=dataset)

    # Resource estimate
    resources = trainer.estimate_resources()
    print(f"\nResource estimates:")
    print(f"  GPU memory: ~{resources['per_gpu_memory_gb']} GB per GPU")
    print(f"  Recommended: {resources['recommended_gpu']}")
    print(f"  Estimated time: ~{resources['estimated_hours']:.1f} hours")
    print(f"  Training samples: {len(dataset)}")
    print(f"  Validation samples: {len(dataset.val_samples)}")

    if args.dry_run:
        print(f"\n--- DRY RUN: Training command ---")
        output_path = trainer.prepare_data_only()
        print(f"\nData prepared at: {output_path}")
        print(f"Training command saved to: {args.output_dir}/train_command.sh")
        print(f"\nTo run manually:")
        print(f"  bash {args.output_dir}/train_command.sh")
        return

    # Confirm
    print(f"\nReady to train:")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA rank: {config.lora_rank}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  LR: {config.learning_rate}")
    try:
        response = input("\nProceed? [y/N] ").strip().lower()
    except (EOFError, OSError):
        # Default to 'y' when running with nohup or no stdin
        print("Proceeding (running in background mode)")
        response = "y"
    if response != "y":
        print("Aborted.")
        sys.exit(0)


    # Train
    result = trainer.train()
    logger.info(f"Training result: {result}")

    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"  Status: {result.get('status', 'unknown')}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}")
    print(f"\nNext: python step3_inference.py --model_path {args.output_dir}")


if __name__ == "__main__":
    main()