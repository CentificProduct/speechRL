"""
RAVDESS Dataset Loader
=======================
Ryerson Audio-Visual Database of Emotional Speech and Song.

- Size: 7,356 recordings (speech portion: 1,440 files)
- Labels: 8 emotions x 2 intensities (normal, strong)
- Role: Controlled ablation studies
- Structure: Actor-based, filename-encoded metadata

Filename convention (speech):
    03-01-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav

    Modality:   01=full-AV, 02=video-only, 03=audio-only
    Channel:    01=speech, 02=song
    Emotion:    01=neutral, 02=calm, 03=happy, 04=sad,
                05=angry, 06=fearful, 07=disgust, 08=surprised
    Intensity:  01=normal, 02=strong
    Statement:  01="Kids are talking", 02="Dogs are sitting"
    Repetition: 01=1st, 02=2nd
    Actor:      01-24 (odd=male, even=female)

Directory structure expected:
    RAVDESS/
    +-- Audio_Speech_Actors_01-24/
        +-- Actor_01/
        +-- Actor_02/
        +-- ...
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from pathlib import Path
from typing import Optional

from dataloader.base import (
    BaseEmotionDataset,
    CategoricalEmotion,
    DatasetSplit,
    EmotionSample,
    LABEL_MAPS,
)

logger = logging.getLogger(__name__)

RAVDESS_STATEMENTS = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}

INTENSITY_MAP = {
    "01": "normal",
    "02": "strong",
}

# Default actor-based splits
DEFAULT_ACTOR_SPLITS = {
    DatasetSplit.TRAIN: list(range(1, 21)),
    DatasetSplit.VALIDATION: [21, 22],
    DatasetSplit.TEST: [23, 24],
}


class RAVDESSDataset(BaseEmotionDataset):
    """
    Loader for the RAVDESS speech emotion database.

    Parameters
    ----------
    root_dir : str or Path
        Path to RAVDESS/ directory.
    split : DatasetSplit
        Which split to load.
    actors : list of int, optional
        Override which actors to include.
    speech_only : bool
        Only load speech files, not song (default: True).
    include_neutral : bool
        Include neutral emotion samples (default: True).
    max_samples : int, optional
        Cap number of samples.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: DatasetSplit = DatasetSplit.TRAIN,
        actors: Optional[list[int]] = None,
        speech_only: bool = True,
        include_neutral: bool = True,
        max_samples: Optional[int] = None,
    ):
        super().__init__(root_dir, split, max_samples)
        self.actors = actors or DEFAULT_ACTOR_SPLITS.get(split, list(range(1, 25)))
        self.speech_only = speech_only
        self.include_neutral = include_neutral

    def _load_samples(self) -> list[EmotionSample]:
        samples = []

        audio_root = self._find_audio_root()
        if audio_root is None:
            logger.error(f"Cannot find RAVDESS audio files in {self.root_dir}")
            return samples

        for actor_num in self.actors:
            actor_dir = audio_root / f"Actor_{actor_num:02d}"
            if not actor_dir.exists():
                actor_dir = audio_root / f"Actor_{actor_num}"
                if not actor_dir.exists():
                    logger.debug(f"Actor directory not found: Actor_{actor_num:02d}")
                    continue

            for wav_file in sorted(actor_dir.glob("*.wav")):
                sample = self._parse_filename(wav_file, actor_num)
                if sample is None:
                    continue
                if self.speech_only and sample.metadata.get("channel") != "speech":
                    continue
                if not self.include_neutral and sample.categorical_emotion == CategoricalEmotion.NEUTRAL:
                    continue
                samples.append(sample)

        logger.info(
            f"RAVDESS: loaded {len(samples)} recordings from "
            f"{len(self.actors)} actors"
        )
        return samples

    def _find_audio_root(self) -> Optional[Path]:
        """Locate the directory containing Actor_XX subdirectories."""
        if (self.root_dir / "Actor_01").exists():
            return self.root_dir

        for subdir in self.root_dir.iterdir():
            if subdir.is_dir():
                if (subdir / "Actor_01").exists():
                    return subdir

        return None

    def _parse_filename(
        self, wav_path: Path, actor_num: int
    ) -> Optional[EmotionSample]:
        """Parse RAVDESS filename to extract all metadata."""
        stem = wav_path.stem
        parts = stem.split("-")

        if len(parts) != 7:
            logger.debug(f"Unexpected filename format: {stem}")
            return None

        modality = parts[0]
        channel = parts[1]
        emotion_code = parts[2]
        intensity_code = parts[3]
        statement_code = parts[4]
        repetition = parts[5]
        actor_code = parts[6]

        if modality == "02":  # Video-only
            return None

        channel_name = "speech" if channel == "01" else "song"
        emotion = LABEL_MAPS["ravdess"].get(emotion_code, CategoricalEmotion.OTHER)
        intensity = INTENSITY_MAP.get(intensity_code, "normal")
        gender = "M" if actor_num % 2 == 1 else "F"
        transcript = RAVDESS_STATEMENTS.get(statement_code, "")

        return EmotionSample(
            utterance_id=stem,
            dataset_name="ravdess",
            split=self.split,
            audio_path=str(wav_path),
            speaker_id=f"Actor_{actor_num:02d}",
            gender=gender,
            transcript=transcript,
            categorical_emotion=emotion,
            emotion_raw_label=emotion_code,
            emotion_intensity=intensity,
            metadata={
                "modality": modality,
                "channel": channel_name,
                "intensity_code": intensity_code,
                "statement_code": statement_code,
                "repetition": repetition,
                "actor_num": actor_num,
            },
        )

    def get_controlled_pairs(
        self, emotion_a: CategoricalEmotion, emotion_b: CategoricalEmotion
    ) -> list[tuple[EmotionSample, EmotionSample]]:
        """
        Get matched pairs for controlled ablation comparison.

        Returns pairs from the same actor, same statement, same repetition,
        differing only in emotion. This is RAVDESS's key advantage for
        ablation studies.
        """
        index: dict[tuple, dict] = {}
        for sample in self._samples:
            key = (
                sample.speaker_id,
                sample.metadata.get("statement_code"),
                sample.metadata.get("repetition"),
                sample.metadata.get("intensity_code"),
            )
            emo = sample.categorical_emotion
            index.setdefault(key, {})[emo] = sample

        pairs = []
        for key, emo_dict in index.items():
            if emotion_a in emo_dict and emotion_b in emo_dict:
                pairs.append((emo_dict[emotion_a], emo_dict[emotion_b]))

        return pairs

    def get_intensity_pairs(
        self, emotion: CategoricalEmotion
    ) -> list[tuple[EmotionSample, EmotionSample]]:
        """
        Get matched normal/strong intensity pairs for the same emotion.

        Useful for studying how emotional intensity maps to acoustic features.
        """
        index: dict[tuple, dict] = {}
        for sample in self._samples:
            if sample.categorical_emotion != emotion:
                continue
            key = (
                sample.speaker_id,
                sample.metadata.get("statement_code"),
                sample.metadata.get("repetition"),
            )
            intensity = sample.emotion_intensity
            index.setdefault(key, {})[intensity] = sample

        pairs = []
        for key, int_dict in index.items():
            if "normal" in int_dict and "strong" in int_dict:
                pairs.append((int_dict["normal"], int_dict["strong"]))

        return pairs