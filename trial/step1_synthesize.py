#!/usr/bin/env python3
"""
Emotion-GSRM: Step 1 — CoT Synthesis with RAVDESS
=====================================================
Generates chain-of-thought training data using GPT-4o as teacher.

Usage:
    export OPENAI_API_KEY=sk-...
    python step1_synthesize.py --meld_dir /path/to/MELD.Raw

Options:
    --max_samples N     Number of samples to synthesize (default: 100 for testing)
    --batch             Use batch evidence mode (1 API call per sample instead of 7)
    --output_dir DIR    Output directory (default: ./meld_cot_output)

Cost estimate:
    ~100 samples with per-dimension evidence = ~700 Stage 1 calls + 100 Stage 2 calls
    At GPT-4o pricing (~$2.50/1M input, $10/1M output), expect ~$5-10 for 100 samples
    Use --batch mode to cut Stage 1 cost by ~5x
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from dataloader.meld import MELDDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/home/azureuser/atik/speechRL_backup/meld_cot_output/step1_synthesis.log"),
    ],
)
logger = logging.getLogger("step1")


def main():
    parser = argparse.ArgumentParser(description="CoT synthesis with RAVDESS")
    parser.add_argument("--meld_dir", type=str, default='/home/azureuser/atik/speechRL_backup/datasets/MELD.Raw')  # Update to your MELD path
    parser.add_argument("--max_samples", type=int, default=1280)
    parser.add_argument("--batch", action="store_true", help="Batch evidence mode")
    parser.add_argument("--output_dir", type=str, default="/home/azureuser/atik/speechRL_backup/meld_cot_output")
    parser.add_argument("--api_key", type=str, default="ollama")
    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable or use --api_key")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    # Import pipeline components
    from dataloader.ravdess import RAVDESSDataset
    from dataloader.base import DatasetSplit
    from cot.pipeline import CoTSynthesisPipeline, PipelineConfig

    # Load MELD
    logger.info(f"Loading MELD from {args.meld_dir}")
    dataset = MELDDataset(
        root_dir=args.meld_dir,
        split=DatasetSplit.TRAIN,
    ).load()
    logger.info(f"Loaded {len(dataset)} samples")
    logger.info(f"Emotion distribution: {dataset.emotion_distribution}")

    # Configure pipeline
    config = PipelineConfig(
        api_key=api_key,
        # teacher_model="gpt-4o",
        teacher_model="mistral",
        stage1_temperature=0.3,
        stage2_temperature=0.4,
        batch_evidence=args.batch,
        verify_scores=True,
        score_tolerance=0.5,
        output_dir=args.output_dir,
        target_samples=args.max_samples,
        checkpoint_every=20,
        api_rate_limit=0.2,  # Be gentle with rate limits
    )

    # Estimate cost
    if args.batch:
        est_calls = args.max_samples * 2  # 1 Stage 1 + 1 Stage 2
    else:
        est_calls = args.max_samples * 8  # 7 Stage 1 + 1 Stage 2
    est_cost = est_calls * 0.01  # Rough $0.01 per call estimate
    logger.info(f"Estimated API calls: ~{est_calls}")
    logger.info(f"Estimated cost: ~${est_cost:.2f}")

    # Confirm
    print(f"\nReady to synthesize {args.max_samples} samples")
    print(f"Estimated API calls: ~{est_calls}")
    print(f"Estimated cost: ~${est_cost:.2f}")
    print(f"Output dir: {args.output_dir}")
    try:
        response = input("\nProceed? [y/N] ").strip().lower()
    except (EOFError, OSError):
        # Default to 'y' when running with nohup or no stdin
        print("Proceeding (running in background mode)")
        response = "y"
    if response != "y":
        print("Aborted.")
        sys.exit(0)

    # Run synthesis
    pipeline = CoTSynthesisPipeline(config)
    start = time.time()
    samples = pipeline.run(dataset)
    elapsed = time.time() - start

    logger.info(f"Synthesis complete: {len(samples)} samples in {elapsed/60:.1f} min")
    logger.info(f"Output saved to: {args.output_dir}")

    print(f"\n{'='*60}")
    print(f"  Synthesis Complete")
    print(f"  Samples: {len(samples)}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Output: {args.output_dir}/")
    print(f"    cot_training_data.jsonl  — full CoT samples")
    print(f"    sft_training_data.jsonl  — SFT-ready format")
    print(f"    synthesis_stats.json     — statistics")
    print(f"{'='*60}")
    print(f"\nNext: python step2_train.py --data_path {args.output_dir}/sft_training_data.jsonl")


if __name__ == "__main__":
    main()