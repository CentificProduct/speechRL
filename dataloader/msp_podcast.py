"""
MSP-Podcast Dataset Loader
============================
MSP-Podcast corpus of naturalistic emotional speech from podcasts.

- Size: ~100 hours, 60K+ segments
- Labels: Continuous VAD (1-7 scale), 5+ annotators per segment
- Role: Supplementary training + out-of-domain (OOD) test set
- Key feature: High annotator count enables reliable inter-rater analysis

Directory structure expected:
    MSP-Podcast/
    +-- Labels/
    |   +-- labels_consensus.csv     # Consensus labels (emotion, VAD)
    |   +-- labels_detailed.csv      # Per-annotator labels (optional)
    +-- Audios/
    |   +-- MSP-PODCAST_0001.wav
    |   +-- MSP-PODCAST_0002.wav
    |   +-- ...
    +-- Partitions/
        +-- partition.txt            # Train/val/test split assignments
"""

import csv
import logging
from pathlib import Path
from typing import Optional

from emotion_gsrm.data.base import (
    BaseEmotionDataset,
    CategoricalEmotion,
    DatasetSplit,
    EmotionSample,
    LABEL_MAPS,
    VADAnnotation,
)

logger = logging.getLogger(__name__)

# MSP-Podcast uses 1-7 scale for VAD
MSP_VAD_MAX = 7.0

# Split name mapping
SPLIT_MAP = {
    DatasetSplit.TRAIN: {"Train"},
    DatasetSplit.VALIDATION: {"Validation", "Development"},
    DatasetSplit.TEST: {"Test1", "Test2", "Test"},
}


class MSPPodcastDataset(BaseEmotionDataset):
    """
    Loader for the MSP-Podcast emotion speech corpus.

    Parameters
    ----------
    root_dir : str or Path
        Path to MSP-Podcast/ directory.
    split : DatasetSplit
        Which split to load.
    normalize_vad_to_5 : bool
        Convert VAD from 1-7 to 1-5 scale (default: True).
    min_annotators : int
        Minimum number of annotators required (default: 3).
    load_detailed_annotations : bool
        Load per-annotator labels if available (default: False).
    max_samples : int, optional
        Cap number of samples.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: DatasetSplit = DatasetSplit.TRAIN,
        normalize_vad_to_5: bool = True,
        min_annotators: int = 3,
        load_detailed_annotations: bool = False,
        max_samples: Optional[int] = None,
    ):
        super().__init__(root_dir, split, max_samples)
        self.normalize_vad_to_5 = normalize_vad_to_5
        self.min_annotators = min_annotators
        self.load_detailed_annotations = load_detailed_annotations

    def _load_samples(self) -> list[EmotionSample]:
        # Load partition assignments
        partitions = self._load_partitions()

        # Load consensus labels
        labels = self._load_consensus_labels()

        # Optionally load detailed per-annotator labels
        detailed = {}
        if self.load_detailed_annotations:
            detailed = self._load_detailed_labels()

        # Build samples
        samples = []
        target_splits = SPLIT_MAP.get(self.split, set())

        for utt_id, partition in partitions.items():
            if partition not in target_splits:
                continue

            if utt_id not in labels:
                continue

            label_info = labels[utt_id]

            # Check annotator count
            n_annotators = label_info.get("n_annotators", 0)
            if n_annotators < self.min_annotators:
                continue

            # Map categorical emotion
            emo_raw = label_info.get("emotion_class", "O")
            emotion = LABEL_MAPS["msp_podcast"].get(
                emo_raw, CategoricalEmotion.OTHER
            )

            # Build VAD annotation
            vad = VADAnnotation(
                valence=label_info.get("valence", 4.0),
                arousal=label_info.get("arousal", 4.0),
                dominance=label_info.get("dominance", 4.0),
                annotator_id="consensus",
            )
            if self.normalize_vad_to_5:
                vad = vad.normalize_to_5(original_max=MSP_VAD_MAX)

            vad_list = [vad]

            # Add per-annotator VADs if available
            if utt_id in detailed:
                for ann in detailed[utt_id]:
                    ann_vad = VADAnnotation(
                        valence=ann["valence"],
                        arousal=ann["arousal"],
                        dominance=ann["dominance"],
                        annotator_id=ann.get("annotator_id", ""),
                    )
                    if self.normalize_vad_to_5:
                        ann_vad = ann_vad.normalize_to_5(original_max=MSP_VAD_MAX)
                    vad_list.append(ann_vad)

            # Resolve audio path
            audio_path = self.root_dir / "Audios" / f"{utt_id}.wav"

            # Extract speaker ID from filename if encoded
            speaker_id = label_info.get("speaker_id", "")
            gender = label_info.get("gender", "")

            sample = EmotionSample(
                utterance_id=utt_id,
                dataset_name="msp_podcast",
                split=self.split,
                audio_path=str(audio_path),
                speaker_id=speaker_id,
                gender=gender,
                categorical_emotion=emotion,
                emotion_raw_label=emo_raw,
                vad_annotations=vad_list,
                metadata={
                    "n_annotators": n_annotators,
                    "partition": partition,
                    "agreement": label_info.get("agreement", None),
                },
            )
            samples.append(sample)

        logger.info(
            f"MSP-Podcast: loaded {len(samples)} segments for "
            f"split={self.split.value}"
        )
        return samples

    def _load_partitions(self) -> dict[str, str]:
        """Load train/val/test partition assignments."""
        partitions = {}

        # Try multiple common partition file locations
        partition_paths = [
            self.root_dir / "Partitions" / "partition.txt",
            self.root_dir / "Partitions" / "Partitions.txt",
            self.root_dir / "partition.txt",
        ]

        partition_file = None
        for p in partition_paths:
            if p.exists():
                partition_file = p
                break

        if partition_file is None:
            logger.warning("No partition file found, assigning all to train")
            return partitions

        with open(partition_file, "r") as f:
            for line in f:
                parts = line.strip().split(";")
                if len(parts) < 2:
                    parts = line.strip().split(",")
                if len(parts) < 2:
                    parts = line.strip().split()
                if len(parts) >= 2:
                    utt_id = parts[0].strip()
                    partition = parts[1].strip()
                    partitions[utt_id] = partition

        return partitions

    def _load_consensus_labels(self) -> dict[str, dict]:
        """Load consensus emotion labels (categorical + VAD)."""
        labels = {}

        label_paths = [
            self.root_dir / "Labels" / "labels_consensus.csv",
            self.root_dir / "Labels" / "labels_consensus.tsv",
            self.root_dir / "labels_consensus.csv",
        ]

        label_file = None
        for p in label_paths:
            if p.exists():
                label_file = p
                break

        if label_file is None:
            logger.warning("No consensus label file found")
            return labels

        delimiter = "\t" if label_file.suffix == ".tsv" else ","

        with open(label_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                utt_id = row.get("FileName", row.get("filename", "")).strip()
                if not utt_id:
                    continue

                # Remove .wav extension if present
                utt_id = utt_id.replace(".wav", "")

                labels[utt_id] = {
                    "emotion_class": row.get(
                        "EmoClass", row.get("emotion", "O")
                    ).strip(),
                    "valence": self._safe_float(
                        row.get("Valence", row.get("valence")), 4.0
                    ),
                    "arousal": self._safe_float(
                        row.get("Arousal", row.get("arousal")), 4.0
                    ),
                    "dominance": self._safe_float(
                        row.get("Dominance", row.get("dominance")), 4.0
                    ),
                    "n_annotators": int(
                        self._safe_float(
                            row.get("NumAnnotators", row.get("n_annotators")), 0
                        )
                    ),
                    "speaker_id": row.get(
                        "SpkrID", row.get("speaker_id", "")
                    ).strip(),
                    "gender": row.get("Gender", row.get("gender", "")).strip(),
                }

        return labels

    def _load_detailed_labels(self) -> dict[str, list[dict]]:
        """Load per-annotator labels if available."""
        detailed = {}

        detail_paths = [
            self.root_dir / "Labels" / "labels_detailed.csv",
            self.root_dir / "Labels" / "labels_detailed.tsv",
        ]

        detail_file = None
        for p in detail_paths:
            if p.exists():
                detail_file = p
                break

        if detail_file is None:
            return detailed

        delimiter = "\t" if detail_file.suffix == ".tsv" else ","

        with open(detail_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                utt_id = row.get("FileName", "").strip().replace(".wav", "")
                if not utt_id:
                    continue

                ann = {
                    "annotator_id": row.get("WorkerID", row.get("annotator", "")),
                    "valence": self._safe_float(row.get("Valence"), 4.0),
                    "arousal": self._safe_float(row.get("Arousal"), 4.0),
                    "dominance": self._safe_float(row.get("Dominance"), 4.0),
                }
                detailed.setdefault(utt_id, []).append(ann)

        return detailed

    @staticmethod
    def _safe_float(val, default: float = 0.0) -> float:
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default