"""
Emotion-GSRM: End-to-End CoT Synthesis Pipeline
=================================================
Orchestrates the full two-stage synthesis process:

  1. Extract acoustic features for each utterance
  2. Run Stage 1 (evidence generation) via GPT-4o
  3. Derive oracle scores from dataset annotations
  4. Run Stage 2 (judgment synthesis) via GPT-4o
  5. Export training samples for SFT

Targets 5K-7K training samples as specified in the proposal.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dataloader.base import EmotionSample, BaseEmotionDataset
from features.acoustic import (
    AcousticFeatureExtractor,
    AcousticFeatureSet,
    FeatureNormalizer,
    format_features_for_prompt,
)
from rubric.init import DimensionName, EmotionScores
from cot.evidence import EvidenceGenerator, EvidenceLog
from cot.judgment import JudgmentSynthesizer, CoTSample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Oracle Score Derivation
# ---------------------------------------------------------------------------

def derive_oracle_scores(sample: EmotionSample) -> dict[str, float]:
    """
    Derive oracle scores from dataset annotations.

    Maps dataset-provided labels (categorical emotion + VAD) to the
    7-dimension evaluation rubric scores. This is a heuristic mapping
    used during CoT synthesis — the teacher model will generate
    reasoning that arrives at these scores naturally.

    Parameters
    ----------
    sample : EmotionSample
        Sample with categorical emotion and/or VAD annotations.

    Returns
    -------
    dict
        Dimension name -> oracle score (1-5 scale).
    """
    scores = {}

    # Start with defaults
    for dim in DimensionName:
        scores[dim.value] = 3.0

    vad = sample.mean_vad

    if vad:
        # Valence accuracy: directly from valence annotation
        scores["valence_accuracy"] = _clamp(vad.valence)

        # Arousal accuracy: directly from arousal annotation
        scores["arousal_accuracy"] = _clamp(vad.arousal)

        # Emotional intensity: correlated with arousal magnitude
        arousal_magnitude = abs(vad.arousal - 3.0)  # Distance from neutral
        scores["emotional_intensity"] = _clamp(
            2.0 + arousal_magnitude * 1.5
        )

    # Use emotion category for appropriateness/consistency heuristics
    if sample.categorical_emotion:
        emo = sample.categorical_emotion.value

        # For acted/controlled datasets (RAVDESS), high appropriateness
        if sample.dataset_name == "ravdess":
            scores["emotional_appropriateness"] = 4.0
            scores["emotional_consistency"] = 4.0
            # Use intensity annotation if available
            if sample.emotion_intensity == "strong":
                scores["emotional_intensity"] = min(
                    scores["emotional_intensity"] + 1.0, 5.0
                )

        # For conversational datasets, context-dependent
        elif sample.dataset_name in ("iemocap", "meld"):
            # Default moderate appropriateness for natural speech
            scores["emotional_appropriateness"] = 3.5
            scores["emotional_consistency"] = 3.5

    # Transition smoothness: based on context availability
    if sample.context_turns:
        # With context, assume moderate smoothness
        scores["transition_smoothness"] = 3.5
    else:
        # Without context, default to neutral
        scores["transition_smoothness"] = 3.0

    # Overall: weighted average of other dimensions
    sub_scores = [
        scores[d.value] for d in DimensionName
        if d != DimensionName.OVERALL_EMOTIONAL_QUALITY
    ]
    scores["overall_emotional_quality"] = _clamp(
        sum(sub_scores) / len(sub_scores)
    )

    # Round to 0.5 increments for cleaner training targets
    for dim in scores:
        scores[dim] = round(scores[dim] * 2) / 2

    return scores


def _clamp(val: float, lo: float = 1.0, hi: float = 5.0) -> float:
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# Pipeline Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for the CoT synthesis pipeline."""

    # API settings
    api_key: str = ""
    teacher_model: str = "gpt-4o"
    stage1_temperature: float = 0.3
    stage2_temperature: float = 0.4

    # Feature extraction
    sample_rate: int = 16000
    min_pitch: float = 75.0
    max_pitch: float = 500.0

    # Pipeline behavior
    batch_evidence: bool = False  # Single-call vs per-dimension evidence
    verify_scores: bool = True    # Retry if scores don't match oracle
    score_tolerance: float = 0.5  # Max deviation from oracle per dimension
    max_verify_attempts: int = 3

    # Parallelism
    n_workers: int = 4            # Parallel feature extraction threads
    api_rate_limit: float = 0.1   # Min seconds between API calls

    # Output
    output_dir: str = "./cot_synthesis_output"
    checkpoint_every: int = 100   # Save checkpoint every N samples
    target_samples: int = 6000    # Target 5K-7K as per proposal

    # Filtering
    min_duration_sec: float = 0.5
    max_duration_sec: float = 30.0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class CoTSynthesisPipeline:
    """
    End-to-end pipeline for CoT training data synthesis.

    Orchestrates acoustic feature extraction, Stage 1 evidence generation,
    oracle score derivation, and Stage 2 judgment synthesis.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.feature_extractor = AcousticFeatureExtractor(
            sample_rate=config.sample_rate,
            min_pitch=config.min_pitch,
            max_pitch=config.max_pitch,
        )
        self.normalizer = FeatureNormalizer()

        self.evidence_generator = EvidenceGenerator(
            api_key=config.api_key,
            model=config.teacher_model,
            temperature=config.stage1_temperature,
            rate_limit_delay=config.api_rate_limit,
        )

        self.judgment_synthesizer = JudgmentSynthesizer(
            api_key=config.api_key,
            model=config.teacher_model,
            temperature=config.stage2_temperature,
        )

        # Tracking
        self._processed = 0
        self._failed = 0
        self._samples: list[CoTSample] = []

    def run(
        self,
        dataset: BaseEmotionDataset,
        resume_from: Optional[str] = None,
    ) -> list[CoTSample]:
        """
        Run the full synthesis pipeline on a dataset.

        Parameters
        ----------
        dataset : BaseEmotionDataset
            Loaded dataset to synthesize training data from.
        resume_from : str, optional
            Path to checkpoint file to resume from.

        Returns
        -------
        list of CoTSample
            Synthesized training samples.
        """
        logger.info(
            f"Starting CoT synthesis pipeline: "
            f"{len(dataset)} samples available, "
            f"target={self.config.target_samples}"
        )

        # Resume from checkpoint if available
        start_idx = 0
        if resume_from and Path(resume_from).exists():
            self._load_checkpoint(resume_from)
            start_idx = self._processed
            logger.info(f"Resumed from checkpoint: {start_idx} samples done")

        # Step 1: Fit normalizer on speaker data
        logger.info("Fitting speaker-level normalizer...")
        self._fit_normalizer(dataset)

        # Step 2: Process each sample through the pipeline
        samples_iter = list(dataset)[start_idx:]
        total = min(len(samples_iter), self.config.target_samples - len(self._samples))

        logger.info(f"Processing {total} samples...")
        start_time = time.time()

        for i, sample in enumerate(samples_iter[:total]):
            try:
                cot_sample = self._process_single(sample)
                if cot_sample is not None:
                    self._samples.append(cot_sample)

                self._processed += 1

                # Progress logging
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    eta = (total - i - 1) / max(rate, 0.01)
                    logger.info(
                        f"Progress: {i + 1}/{total} "
                        f"({len(self._samples)} good, {self._failed} failed) "
                        f"[{rate:.1f} samples/s, ETA: {eta / 60:.0f}min]"
                    )

                # Checkpoint
                if (i + 1) % self.config.checkpoint_every == 0:
                    self._save_checkpoint()

            except Exception as e:
                logger.error(
                    f"Failed to process {sample.utterance_id}: {e}",
                    exc_info=True,
                )
                self._failed += 1

        # Final save
        self._save_output()

        elapsed = time.time() - start_time
        logger.info(
            f"Pipeline complete: {len(self._samples)} samples synthesized "
            f"in {elapsed / 60:.1f} minutes "
            f"({self._failed} failures)"
        )

        return self._samples

    def _process_single(self, sample: EmotionSample) -> Optional[CoTSample]:
        """
        Process a single sample through the full pipeline.

        Steps:
          1. Extract acoustic features
          2. Normalize and discretize
          3. Generate evidence log (Stage 1)
          4. Derive oracle scores
          5. Synthesize judgment CoT (Stage 2)
        """
        # Duration filtering
        if sample.duration_sec > 0:
            if sample.duration_sec < self.config.min_duration_sec:
                logger.debug(f"Skipping {sample.utterance_id}: too short")
                return None
            if sample.duration_sec > self.config.max_duration_sec:
                logger.debug(f"Skipping {sample.utterance_id}: too long")
                return None

        # Step 1: Extract acoustic features
        try:
            feature_set = self.feature_extractor.extract(
                audio_path=sample.audio_path,
                utterance_id=sample.utterance_id,
                speaker_id=sample.speaker_id,
            )
        except Exception as e:
            logger.warning(
                f"Feature extraction failed for {sample.utterance_id}: {e}"
            )
            self._failed += 1
            return None

        # Step 2: Normalize and discretize
        normalized = self.normalizer.normalize(feature_set)
        discretized = self.normalizer.discretize(normalized)

        # Step 3: Format features for prompt
        features_text = format_features_for_prompt(feature_set, discretized)

        # Step 4: Generate evidence log (Stage 1)
        if self.config.batch_evidence:
            evidence_log = self.evidence_generator.generate_evidence_batch(
                sample, features_text
            )
        else:
            evidence_log = self.evidence_generator.generate_evidence(
                sample, features_text
            )

        # Store audio path in metadata for training format
        evidence_log.generation_metadata["audio_path"] = sample.audio_path

        # Step 5: Derive oracle scores
        oracle_scores = derive_oracle_scores(sample)

        # Step 6: Synthesize judgment CoT (Stage 2)
        if self.config.verify_scores:
            cot_sample = self.judgment_synthesizer.synthesize_with_verification(
                evidence_log=evidence_log,
                oracle_scores=oracle_scores,
                max_attempts=self.config.max_verify_attempts,
                tolerance=self.config.score_tolerance,
            )
        else:
            cot_sample = self.judgment_synthesizer.synthesize(
                evidence_log=evidence_log,
                oracle_scores=oracle_scores,
            )

        return cot_sample

    def _fit_normalizer(self, dataset: BaseEmotionDataset) -> None:
        """
        Fit the feature normalizer on a subset of speaker data.

        For efficiency, extracts features from a small sample per speaker
        to compute normalization statistics.
        """
        speakers = dataset.speakers
        if not speakers:
            logger.warning("No speakers found, skipping normalizer fitting")
            return

        samples_per_speaker = 5  # Small sample for stats
        all_features = []

        for speaker_id in speakers[:50]:  # Cap at 50 speakers for speed
            speaker_samples = dataset.filter_by_speaker(speaker_id)
            speaker_features = []

            for sample in speaker_samples[:samples_per_speaker]:
                try:
                    fs = self.feature_extractor.extract(
                        audio_path=sample.audio_path,
                        utterance_id=sample.utterance_id,
                        speaker_id=speaker_id,
                    )
                    speaker_features.append(fs)
                    all_features.append(fs)
                except Exception:
                    continue

            if speaker_features:
                self.normalizer.fit_speaker(speaker_id, speaker_features)

        if all_features:
            self.normalizer.fit_quantiles(all_features)
            logger.info(
                f"Normalizer fitted on {len(all_features)} samples "
                f"from {len(speakers)} speakers"
            )

    def _save_checkpoint(self) -> None:
        """Save a pipeline checkpoint for resumability."""
        checkpoint_path = self.output_dir / "checkpoint.jsonl"
        with open(checkpoint_path, "w") as f:
            for sample in self._samples:
                f.write(json.dumps(sample.to_dict()) + "\n")

        meta_path = self.output_dir / "checkpoint_meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "processed": self._processed,
                "failed": self._failed,
                "n_samples": len(self._samples),
            }, f)

        logger.info(
            f"Checkpoint saved: {len(self._samples)} samples "
            f"at {checkpoint_path}"
        )

    def _load_checkpoint(self, path: str) -> None:
        """Load a pipeline checkpoint."""
        checkpoint_path = Path(path)
        if checkpoint_path.suffix == ".json":
            # Load metadata
            meta_path = checkpoint_path
            data_path = checkpoint_path.parent / "checkpoint.jsonl"
        else:
            data_path = checkpoint_path
            meta_path = checkpoint_path.parent / "checkpoint_meta.json"

        if data_path.exists():
            with open(data_path) as f:
                for line in f:
                    if line.strip():
                        sample = CoTSample.from_dict(json.loads(line))
                        self._samples.append(sample)

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                self._processed = meta.get("processed", len(self._samples))
                self._failed = meta.get("failed", 0)

    def _save_output(self) -> None:
        """Save final synthesized training data."""
        # JSONL format (one sample per line)
        output_path = self.output_dir / "cot_training_data.jsonl"
        with open(output_path, "w") as f:
            for sample in self._samples:
                f.write(json.dumps(sample.to_dict()) + "\n")

        # SFT-ready format
        sft_path = self.output_dir / "sft_training_data.jsonl"
        with open(sft_path, "w") as f:
            for sample in self._samples:
                f.write(json.dumps(sample.to_training_format()) + "\n")

        # Summary statistics
        stats = self._compute_stats()
        stats_path = self.output_dir / "synthesis_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(
            f"Output saved to {self.output_dir}: "
            f"{len(self._samples)} training samples"
        )

    def _compute_stats(self) -> dict:
        """Compute summary statistics for the synthesized data."""
        import numpy as np

        if not self._samples:
            return {"n_samples": 0}

        # Score distributions
        score_stats = {}
        for dim in DimensionName:
            values = [s.scores.get(dim.value, 3.0) for s in self._samples]
            score_stats[dim.value] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        # Oracle match statistics
        match_rates = []
        for sample in self._samples:
            if sample.oracle_scores:
                matched = sum(
                    1 for d in DimensionName
                    if abs(
                        sample.scores.get(d.value, 3.0)
                        - sample.oracle_scores.get(d.value, 3.0)
                    ) <= self.config.score_tolerance
                )
                match_rates.append(matched / len(DimensionName))

        return {
            "n_samples": len(self._samples),
            "n_processed": self._processed,
            "n_failed": self._failed,
            "success_rate": len(self._samples) / max(self._processed, 1),
            "score_statistics": score_stats,
            "oracle_match_rate": float(np.mean(match_rates)) if match_rates else 0.0,
            "config": {
                "teacher_model": self.config.teacher_model,
                "batch_evidence": self.config.batch_evidence,
                "verify_scores": self.config.verify_scores,
            },
        }