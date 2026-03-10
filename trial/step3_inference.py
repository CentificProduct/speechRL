#!/usr/bin/env python3
"""
Emotion-GSRM: Step 3 — Inference & Evaluation
================================================
Run the trained model on held-out RAVDESS test samples.

Usage:
    python step3_inference.py \
        --model_path ./emotion_gsrm_checkpoints \
        --ravdess_dir /path/to/RAVDESS

Options:
    --k N               Number of samples to average (default: 16)
    --temperature F     Sampling temperature (default: 1.0)
    --top_p F           Top-p sampling (default: 0.6)
    --output_path FILE  Results output (default: ./ravdess_results.jsonl)
    --max_test N        Max test samples (default: 50)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging    
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("step3")


def main():
    parser = argparse.ArgumentParser(description="Emotion-GSRM inference")
    parser.add_argument("--model_path", type=str, 
        default="/home/azureuser/atik/speechRL/emotion_gsrm_checkpoints/v12-20260309-183522/checkpoint-100")
    parser.add_argument("--ravdess_dir", type=str, default="/home/azureuser/atik/speechRL/datasets/RAVDESS")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.6)
    parser.add_argument("--output_path", type=str, default="./ravdess_results.jsonl")
    parser.add_argument("--max_test", type=int, default=50)
    args = parser.parse_args()

    from dataloader.ravdess import RAVDESSDataset
    from dataloader.base import DatasetSplit
    from sft.inference import EmotionGSRMInference
    from rubric.init import DimensionName

    # Load test split
    logger.info("Loading RAVDESS test split (Actors 23-24)")
    dataset = RAVDESSDataset(
        root_dir=args.ravdess_dir,
        split=DatasetSplit.TEST,
    ).load()
    logger.info(f"Test samples: {len(dataset)}")

    test_samples = list(dataset)[:args.max_test]

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    engine = EmotionGSRMInference(
        model_path=args.model_path,
        k=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
    ).load_model()

    # Run inference
    results = []
    start = time.time()

    for i, sample in enumerate(test_samples):
        logger.info(f"[{i+1}/{len(test_samples)}] {sample.utterance_id}")

        result = engine.predict(
            audio_path=sample.audio_path,
            transcript=sample.transcript,
            utterance_id=sample.utterance_id,
        )
        results.append(result)

        # Print progress
        print(f"  {sample.utterance_id}: "
              f"emotion={sample.categorical_emotion.value}, "
              f"intensity={sample.emotion_intensity}")
        for dim in DimensionName:
            score = result.scores[dim.value]
            std = result.score_stds[dim.value]
            print(f"    {dim.value}: {score:.2f} (std={std:.3f})")

    elapsed = time.time() - start

    # Export results
    engine.export_results(results, args.output_path)

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"  Inference Complete")
    print(f"  Samples: {len(results)}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(results):.1f}s per sample)")
    print(f"  K: {args.k} samples averaged per utterance")
    print(f"{'='*60}")

    print(f"\nScore statistics across {len(results)} test samples:")
    for dim in DimensionName:
        scores = [r.scores[dim.value] for r in results]
        stds = [r.score_stds[dim.value] for r in results]
        print(f"  {dim.value}:")
        print(f"    mean={np.mean(scores):.2f}, std={np.std(scores):.2f}, "
              f"avg_k_std={np.mean(stds):.3f}")

    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()