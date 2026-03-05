"""
Emotion-GSRM: Base data structures and abstract dataset class.
================================================================
Defines the common sample format and interface that all four dataset
loaders (IEMOCAP, MSP-Podcast, RAVDESS, MELD) implement.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Emotion Taxonomy
# ---------------------------------------------------------------------------

class CategoricalEmotion(str, Enum):
    """Superset of categorical emotion labels across all four datasets."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    EXCITED = "excited"
    FRUSTRATED = "frustrated"
    CALM = "calm"
    OTHER = "other"


# Mapping from dataset-specific labels to canonical labels
LABEL_MAPS: dict[str, dict[str, CategoricalEmotion]] = {
    "iemocap": {
        "neu": CategoricalEmotion.NEUTRAL,
        "hap": CategoricalEmotion.HAPPY,
        "sad": CategoricalEmotion.SAD,
        "ang": CategoricalEmotion.ANGRY,
        "fru": CategoricalEmotion.FRUSTRATED,
        "exc": CategoricalEmotion.EXCITED,
        "fea": CategoricalEmotion.FEAR,
        "sur": CategoricalEmotion.SURPRISE,
        "dis": CategoricalEmotion.DISGUST,
        "oth": CategoricalEmotion.OTHER,
        "xxx": CategoricalEmotion.OTHER,
    },
    "msp_podcast": {
        "N": CategoricalEmotion.NEUTRAL,
        "H": CategoricalEmotion.HAPPY,
        "S": CategoricalEmotion.SAD,
        "A": CategoricalEmotion.ANGRY,
        "D": CategoricalEmotion.DISGUST,
        "F": CategoricalEmotion.FEAR,
        "U": CategoricalEmotion.SURPRISE,
        "C": CategoricalEmotion.CALM,
        "O": CategoricalEmotion.OTHER,
    },
    "ravdess": {
        "01": CategoricalEmotion.NEUTRAL,
        "02": CategoricalEmotion.CALM,
        "03": CategoricalEmotion.HAPPY,
        "04": CategoricalEmotion.SAD,
        "05": CategoricalEmotion.ANGRY,
        "06": CategoricalEmotion.FEAR,
        "07": CategoricalEmotion.DISGUST,
        "08": CategoricalEmotion.SURPRISE,
    },
    "meld": {
        "neutral": CategoricalEmotion.NEUTRAL,
        "joy": CategoricalEmotion.HAPPY,
        "sadness": CategoricalEmotion.SAD,
        "anger": CategoricalEmotion.ANGRY,
        "fear": CategoricalEmotion.FEAR,
        "disgust": CategoricalEmotion.DISGUST,
        "surprise": CategoricalEmotion.SURPRISE,
    },
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


@dataclass
class VADAnnotation:
    """Valence-Arousal-Dominance dimensional annotation."""
    valence: float       # 1-5 or 1-7 scale (dataset-dependent)
    arousal: float       # 1-5 or 1-7 scale
    dominance: float     # 1-5 or 1-7 scale
    annotator_id: str = ""

    def normalize_to_5(self, original_max: float = 5.0) -> "VADAnnotation":
        """Re-scale VAD values to a 1-5 range if needed."""
        if original_max == 5.0:
            return self
        scale = 4.0 / (original_max - 1.0)  # Maps [1, max] -> [1, 5]
        return VADAnnotation(
            valence=1.0 + (self.valence - 1.0) * scale,
            arousal=1.0 + (self.arousal - 1.0) * scale,
            dominance=1.0 + (self.dominance - 1.0) * scale,
            annotator_id=self.annotator_id,
        )


@dataclass
class EmotionSample:
    """
    A single utterance sample with emotion annotations.

    This is the universal format consumed by the feature extraction
    pipeline and CoT synthesis stages.
    """
    # Identity
    utterance_id: str
    dataset_name: str
    split: DatasetSplit

    # Audio
    audio_path: str
    sample_rate: int = 16000
    duration_sec: float = 0.0

    # Speaker info
    speaker_id: str = ""
    gender: str = ""  # "M", "F", or ""

    # Transcript
    transcript: str = ""

    # Conversational context (preceding turns, if available)
    context_turns: list[str] = field(default_factory=list)

    # Categorical emotion
    categorical_emotion: Optional[CategoricalEmotion] = None
    emotion_raw_label: str = ""  # Original dataset label before mapping

    # Dimensional annotations (VAD)
    vad_annotations: list[VADAnnotation] = field(default_factory=list)

    # Sentiment (MELD-specific, but kept general)
    sentiment: str = ""  # "positive", "negative", "neutral"

    # RAVDESS-specific intensity
    emotion_intensity: str = ""  # "normal" or "strong"

    # Additional metadata
    metadata: dict = field(default_factory=dict)

    @property
    def mean_vad(self) -> Optional[VADAnnotation]:
        """Average VAD across annotators."""
        if not self.vad_annotations:
            return None
        import numpy as np
        return VADAnnotation(
            valence=float(np.mean([a.valence for a in self.vad_annotations])),
            arousal=float(np.mean([a.arousal for a in self.vad_annotations])),
            dominance=float(np.mean([a.dominance for a in self.vad_annotations])),
            annotator_id=f"mean_{len(self.vad_annotations)}",
        )

    @property
    def n_annotators(self) -> int:
        return len(self.vad_annotations)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON export."""
        d = asdict(self)
        d["categorical_emotion"] = (
            self.categorical_emotion.value if self.categorical_emotion else None
        )
        d["split"] = self.split.value
        return d


# ---------------------------------------------------------------------------
# Abstract Base Dataset
# ---------------------------------------------------------------------------

class BaseEmotionDataset(ABC):
    """
    Abstract base class for emotion speech datasets.

    All four dataset loaders implement this interface, providing
    a consistent API for the rest of the pipeline.

    Parameters
    ----------
    root_dir : str or Path
        Root directory of the dataset.
    split : DatasetSplit
        Which split to load.
    max_samples : int, optional
        Cap the number of samples loaded (for debugging).
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: DatasetSplit = DatasetSplit.TRAIN,
        max_samples: Optional[int] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.max_samples = max_samples
        self._samples: list[EmotionSample] = []
        self._loaded = False

    @property
    def name(self) -> str:
        """Dataset name identifier."""
        return self.__class__.__name__.replace("Dataset", "").lower()

    def load(self) -> "BaseEmotionDataset":
        """Load dataset samples. Returns self for chaining."""
        if self._loaded:
            logger.info(f"{self.name} already loaded ({len(self._samples)} samples)")
            return self

        logger.info(f"Loading {self.name} split={self.split.value} from {self.root_dir}")
        self._samples = self._load_samples()

        if self.max_samples and len(self._samples) > self.max_samples:
            self._samples = self._samples[:self.max_samples]

        self._loaded = True
        logger.info(f"Loaded {len(self._samples)} samples from {self.name}")
        return self

    @abstractmethod
    def _load_samples(self) -> list[EmotionSample]:
        """Dataset-specific loading logic. Implemented by subclasses."""
        ...

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> EmotionSample:
        return self._samples[idx]

    def __iter__(self) -> Iterator[EmotionSample]:
        return iter(self._samples)

    def filter_by_emotion(
        self, emotions: list[CategoricalEmotion]
    ) -> list[EmotionSample]:
        """Filter samples to only include specified emotions."""
        return [s for s in self._samples if s.categorical_emotion in emotions]

    def filter_by_speaker(self, speaker_id: str) -> list[EmotionSample]:
        """Get all samples from a specific speaker."""
        return [s for s in self._samples if s.speaker_id == speaker_id]

    @property
    def speakers(self) -> list[str]:
        """Unique speaker IDs in this split."""
        return sorted(set(s.speaker_id for s in self._samples if s.speaker_id))

    @property
    def emotion_distribution(self) -> dict[str, int]:
        """Count of samples per categorical emotion."""
        from collections import Counter
        return dict(Counter(
            s.categorical_emotion.value
            for s in self._samples
            if s.categorical_emotion
        ))

    def get_context_window(
        self, sample: EmotionSample, n_turns: int = 3
    ) -> list[str]:
        """
        Retrieve preceding conversational turns for a sample.

        If the dataset provides dialogue context (IEMOCAP, MELD),
        returns up to n_turns preceding utterance transcripts.
        Otherwise returns sample.context_turns as-is.
        """
        if sample.context_turns:
            return sample.context_turns[-n_turns:]
        return []

    def export_manifest(self, output_path: str | Path) -> None:
        """
        Export dataset manifest as JSONL for downstream consumption.

        Each line is a JSON object representing one EmotionSample.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for sample in self._samples:
                f.write(json.dumps(sample.to_dict()) + "\n")

        logger.info(f"Exported {len(self._samples)} samples to {output_path}")

    def summary(self) -> str:
        """Print a human-readable dataset summary."""
        lines = [
            f"Dataset: {self.name}",
            f"Split: {self.split.value}",
            f"Samples: {len(self._samples)}",
            f"Speakers: {len(self.speakers)}",
            f"Emotion distribution:",
        ]
        for emo, count in sorted(self.emotion_distribution.items()):
            lines.append(f"  {emo}: {count}")

        has_vad = sum(1 for s in self._samples if s.vad_annotations)
        has_transcript = sum(1 for s in self._samples if s.transcript)
        has_context = sum(1 for s in self._samples if s.context_turns)
        lines.append(f"With VAD annotations: {has_vad}")
        lines.append(f"With transcripts: {has_transcript}")
        lines.append(f"With context turns: {has_context}")

        return "\n".join(lines)