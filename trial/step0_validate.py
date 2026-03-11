#!/usr/bin/env python3
"""
Emotion-GSRM: Step 0 — Validate Pipeline with RAVDESS
========================================================
Run this FIRST to verify every component works before spending
money on GPT-4o API calls.

Usage:
    python step0_validate.py --ravdess_dir /path/to/RAVDESS

This script:
  1. Loads RAVDESS data
  2. Extracts acoustic features from a few samples
  3. Tests normalization & discretization
  4. Generates a mock evidence log (no API calls)
  5. Tests oracle score derivation
  6. Tests CoT sample formatting for SFT
  7. Reports any missing dependencies

If this passes, you're ready for Step 1.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import sys
import time
from pathlib import Path


def check_dependencies():
    """Check all required packages are installed."""
    print("=" * 60)
    print("STEP 0.1: Checking dependencies")
    print("=" * 60)

    required = {
        "numpy": "numpy",
        "librosa": "librosa",
        "parselmouth": "parselmouth",
        "soundfile": "soundfile",
    }
    optional = {
        "openai": "openai (needed for Step 1: CoT synthesis)",
        "transformers": "transformers (needed for Step 3: training)",
        "torch": "torch (needed for Step 3: training)",
        "peft": "peft (needed for Step 3: LoRA training)",
    }

    missing_required = []
    missing_optional = []

    for module, name in required.items():
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name}")
            missing_required.append(name)

    for module, name in optional.items():
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [SKIP] {name} — not needed for validation")
            missing_optional.append(name)

    if missing_required:
        print(f"\nERROR: Missing required packages: {missing_required}")
        print("Install with: pip install " + " ".join(missing_required))
        sys.exit(1)

    print(f"\nAll required dependencies OK.")
    if missing_optional:
        print(f"Optional (install later): {missing_optional}")
    print()


def validate_meld_loading(meld_dir: str):
    print("=" * 60)
    print("STEP 0.2: Loading MELD dataset")
    print("=" * 60)

    from dataloader.meld import MELDDataset
    from dataloader.base import DatasetSplit

    dataset = MELDDataset(
        root_dir=meld_dir,
        split=DatasetSplit.TEST,
        require_audio=True,
    ).load()

    print(f"  Loaded {len(dataset)} samples")
    print(f"  Speakers: {len(dataset.speakers)}")
    print(f"  Emotion distribution: {dataset.emotion_distribution}")

    if len(dataset) == 0:
        print("\nERROR: No samples loaded!")
        print(f"Check that {meld_dir} contains test/test_sent_emo.csv and extracted audio")
        sys.exit(1)

    sample = dataset[0]
    print(f"\n  Sample 0:")
    print(f"    ID: {sample.utterance_id}")
    print(f"    Audio: {sample.audio_path}")
    print(f"    Emotion: {sample.categorical_emotion}")
    print(f"    Sentiment: {sample.sentiment}")
    print(f"    Speaker: {sample.speaker_id}")
    print(f"    Transcript: {sample.transcript}")
    print(f"    Context: {sample.context_turns}")

    if Path(sample.audio_path).exists():
        print(f"    Audio file: EXISTS")
    else:
        print(f"    Audio file: NOT FOUND at {sample.audio_path}")
        sys.exit(1)

    print()
    return dataset


def validate_feature_extraction(dataset):
    """Test acoustic feature extraction on a few samples."""
    print("=" * 60)
    print("STEP 0.3: Extracting acoustic features (3 samples)")
    print("=" * 60)

    from features.acoustic import AcousticFeatureExtractor

    extractor = AcousticFeatureExtractor(sample_rate=16000)
    feature_sets = []

    for i, sample in enumerate(dataset):
        if i >= 3:
            break

        start = time.time()
        try:
            fs = extractor.extract(
                audio_path=sample.audio_path,
                utterance_id=sample.utterance_id,
                speaker_id=sample.speaker_id,
            )
            elapsed = time.time() - start
            feature_sets.append(fs)

            print(f"\n  Sample {i}: {sample.utterance_id} ({elapsed:.2f}s)")
            print(f"    Prosodic: pitch_mean={fs.prosodic.pitch_level.mean():.1f} Hz, "
                  f"n_segments={len(fs.prosodic.pitch_level)}")
            print(f"    Voice Quality: jitter={fs.voice_quality.jitter:.3f}%, "
                  f"shimmer={fs.voice_quality.shimmer:.3f}%, "
                  f"HNR={fs.voice_quality.hnr:.1f} dB, "
                  f"spectral_tilt={fs.voice_quality.spectral_tilt:.2f}")
            print(f"    Temporal: rate={fs.temporal.speech_rate:.1f} syl/s, "
                  f"pauses={fs.temporal.pause_frequency:.2f}/s")
            print(f"    Formant: F1={fs.formant.f1_mean:.0f} Hz, "
                  f"F2={fs.formant.f2_mean:.0f} Hz, "
                  f"VSA={fs.formant.vowel_space_area:.0f} Hz^2")

        except Exception as e:
            print(f"\n  Sample {i}: FAILED — {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print()
    return feature_sets


def validate_normalization(feature_sets):
    """Test normalization and discretization."""
    print("=" * 60)
    print("STEP 0.4: Normalizing & discretizing features")
    print("=" * 60)

    from features.acoustic import (
        FeatureNormalizer, format_features_for_prompt,
    )

    normalizer = FeatureNormalizer()

    # Fit on available data
    speaker_id = feature_sets[0].speaker_id or "test_speaker"
    normalizer.fit_speaker(speaker_id, feature_sets)
    normalizer.fit_quantiles(feature_sets)
    print(f"  Fitted normalizer on {len(feature_sets)} samples")

    # Normalize & discretize first sample
    normalized = normalizer.normalize(feature_sets[0])
    discretized = normalizer.discretize(normalized)

    print(f"\n  Normalized features ({len(normalized)} dimensions):")
    for name, val in list(normalized.items())[:5]:
        print(f"    {name}: z={val:.3f} -> {discretized[name]}")
    print(f"    ... ({len(normalized) - 5} more)")

    # Format for prompt
    prompt_text = format_features_for_prompt(feature_sets[0], discretized)
    print(f"\n  Prompt text ({len(prompt_text)} chars):")
    for line in prompt_text.split("\n")[:8]:
        print(f"    {line}")
    print(f"    ... ({len(prompt_text.split(chr(10))) - 8} more lines)")

    print()
    return normalized, discretized, prompt_text


def validate_rubric():
    """Test rubric and prompt building."""
    print("=" * 60)
    print("STEP 0.5: Testing evaluation rubric")
    print("=" * 60)

    from rubric.init import (
        DimensionName, RUBRIC_DIMENSIONS, EmotionScores,
        RubricPromptBuilder, aggregate_scores,
    )

    print(f"  Dimensions: {len(DimensionName)} defined")
    for dim in DimensionName:
        rd = RUBRIC_DIMENSIONS[dim]
        print(f"    {rd.display_name}: {len(rd.anchors)} anchors, "
              f"{len(rd.associated_features)} features")

    # Test EmotionScores
    scores = EmotionScores(4.0, 3.5, 4.0, 4.5, 3.0, 3.5, 4.0)
    print(f"\n  EmotionScores: mean={scores.mean_score():.2f}")
    print(f"  to_dict: {scores.to_dict()}")

    # Test prompt building
    evidence_prompt = RubricPromptBuilder.build_evidence_prompt(
        dimension=DimensionName.EMOTIONAL_INTENSITY,
        acoustic_features_text="pitch_level: high (245 Hz)",
        transcript="I am so happy!",
        context_turns=["[Speaker1] How are you?"],
    )
    print(f"\n  Evidence prompt: {len(evidence_prompt)} chars")

    # Test score parsing
    mock_response = (
        "[SCORES]\n"
        "emotional_intensity: 4.0\n"
        "emotional_appropriateness: 3.5\n"
        "emotional_consistency: 4.0\n"
        "valence_accuracy: 4.5\n"
        "arousal_accuracy: 3.0\n"
        "transition_smoothness: 3.5\n"
        "overall_emotional_quality: 4.0\n"
    )
    parsed = RubricPromptBuilder.parse_scores_from_response(mock_response)
    print(f"  Score parsing: {parsed}")

    print()


def validate_oracle_scores(dataset):
    """Test oracle score derivation."""
    print("=" * 60)
    print("STEP 0.6: Testing oracle score derivation")
    print("=" * 60)

    from cot.pipeline import derive_oracle_scores

    for i, sample in enumerate(dataset):
        if i >= 3:
            break
        oracle = derive_oracle_scores(sample)
        print(f"  {sample.utterance_id} ({sample.categorical_emotion.value}, "
              f"intensity={sample.emotion_intensity}):")
        for dim, score in oracle.items():
            print(f"    {dim}: {score}")
        print()


def validate_cot_sample_format(dataset, feature_sets, discretized, prompt_text):
    """Test the full CoT sample creation (without API calls)."""
    print("=" * 60)
    print("STEP 0.7: Testing CoT sample format (mock, no API)")
    print("=" * 60)

    from cot.evidence import EvidenceLog, DimensionEvidence
    from cot.judgment import CoTSample
    from cot.pipeline import derive_oracle_scores
    from rubric.init import DimensionName

    sample = dataset[0]
    oracle = derive_oracle_scores(sample)

    # Create a mock evidence log
    log = EvidenceLog(
        utterance_id=sample.utterance_id,
        transcript=sample.transcript,
        context_turns=sample.context_turns,
        acoustic_features_text=prompt_text,
    )
    for dim in DimensionName:
        log.dimension_evidence[dim.value] = DimensionEvidence(
            dimension=dim.value,
            emotional_context="[MOCK] Speaker delivering scripted emotional content",
            detected_cues="[MOCK] Pitch variation is high, jitter elevated",
            alignment_assessment="[MOCK] Delivery matches expected emotion",
            strengths="[MOCK] Clear emotional expression",
            issues="[MOCK] None",
        )

    # Create a mock CoT sample
    cot_sample = CoTSample(
        utterance_id=sample.utterance_id,
        evidence_log=log,
        reasoning="[MOCK] The speaker shows clear emotional expression through "
                  "elevated pitch variation and increased jitter, consistent with "
                  "the target emotion...",
        scores=oracle,
        oracle_scores=oracle,
    )

    # Test training format
    training_format = cot_sample.to_training_format()
    print(f"  Training sample keys: {list(training_format.keys())}")
    print(f"  Target output length: {len(training_format['target_output'])} chars")
    print(f"  Has [EVIDENCE]: {'[EVIDENCE]' in training_format['target_output']}")
    print(f"  Has [REASONING]: {'[REASONING]' in training_format['target_output']}")
    print(f"  Has [SCORES]: {'[SCORES]' in training_format['target_output']}")

    # Test serialization roundtrip
    d = cot_sample.to_dict()
    cot_sample2 = CoTSample.from_dict(d)
    assert cot_sample2.utterance_id == cot_sample.utterance_id
    print(f"  Serialization roundtrip: OK")

    print()


def validate_sft_dataset_format():
    """Test SFT dataset formatting."""
    print("=" * 60)
    print("STEP 0.8: Testing SFT dataset formatting")
    print("=" * 60)

    import json
    import tempfile
    from sft.dataset import SFTDataset, SYSTEM_PROMPT

    # Create a mock JSONL file
    mock_data = {
        "utterance_id": "test_001",
        "audio_path": "/path/to/audio.wav",
        "transcript": "Kids are talking by the door.",
        "context_turns": [],
        "target_output": "[EVIDENCE]\nMock evidence\n[REASONING]\nMock reasoning\n[SCORES]\nemotional_intensity: 4.0\nemotional_appropriateness: 3.5\nemotional_consistency: 4.0\nvalence_accuracy: 4.5\narousal_accuracy: 3.0\ntransition_smoothness: 3.5\noverall_emotional_quality: 4.0",
        "scores": {"emotional_intensity": 4.0},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(mock_data) + "\n")
        # Write a few more
        for i in range(4):
            mock_data["utterance_id"] = f"test_{i+2:03d}"
            f.write(json.dumps(mock_data) + "\n")
        tmp_path = f.name

    dataset = SFTDataset(data_path=tmp_path, val_split=0.2).load()
    print(f"  Train samples: {len(dataset.train_samples)}")
    print(f"  Val samples: {len(dataset.val_samples)}")

    if dataset.train_samples:
        s = dataset.train_samples[0]
        print(f"  Message roles: {[m['role'] for m in s['messages']]}")
        print(f"  System prompt starts with: '{SYSTEM_PROMPT[:50]}...'")
        print(f"  Has audio: {bool(s.get('audios'))}")

    # Cleanup
    Path(tmp_path).unlink()
    print()


def validate_training_config():
    """Test training configuration."""
    print("=" * 60)
    print("STEP 0.9: Testing training configuration")
    print("=" * 60)

    from sft.config import TrainingConfig
    from sft.trainer import EmotionGSRMTrainer

    config = TrainingConfig(num_gpus=1, per_device_batch_size=1, gradient_accumulation_steps=32)
    warnings = config.validate()
    print(f"  Config warnings: {len(warnings)}")
    for w in warnings:
        print(f"    {w}")

    cli = config.to_swift_cli()
    print(f"\n  SWIFT CLI command ({len(cli)} chars):")
    for line in cli.split("\n")[:5]:
        print(f"    {line}")

    trainer = EmotionGSRMTrainer(config)
    resources = trainer.estimate_resources()
    print(f"\n  Resource estimates:")
    print(f"    Memory: {resources['total_memory_gb']} GB")
    print(f"    GPU: {resources['recommended_gpu']}")
    print(f"    Steps: {resources['estimated_steps']}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Validate Emotion-GSRM pipeline with MELD data"
    )
    parser.add_argument(
        "--meld_dir",
        type=str,
        # required=True,
        default='/home/azureuser/atik/speechRL_backup/datasets/MELD.Raw',
        help="Path to MELD dataset directory (containing subdirs)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Emotion-GSRM Pipeline Validation")
    print("  Using MELD dataset")
    print("=" * 60 + "\n")

    start = time.time()

    # Run all validation steps
    check_dependencies()
    dataset = validate_meld_loading(args.meld_dir)
    feature_sets = validate_feature_extraction(dataset)
    normalized, discretized, prompt_text = validate_normalization(feature_sets)
    validate_rubric()
    validate_oracle_scores(dataset)
    validate_cot_sample_format(dataset, feature_sets, discretized, prompt_text)
    validate_sft_dataset_format()
    validate_training_config()

    elapsed = time.time() - start

    print("=" * 60)
    print(f"  ALL VALIDATION PASSED ({elapsed:.1f}s)")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Set your OpenAI API key: export OPENAI_API_KEY=sk-...")
    print("  2. Run: python step1_synthesize.py --ravdess_dir /path/to/RAVDESS")
    print()


if __name__ == "__main__":
    main()