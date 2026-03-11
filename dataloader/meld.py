"""
MELD Dataset Loader
====================
Multimodal EmotionLines Dataset (from Friends TV series).

- Size: 13K+ utterances
- Labels: 7 emotions + 3-class sentiment
- Role: Out-of-domain (OOD) evaluation for multi-party dialogue
- Key feature: Multi-party conversational context (3+ speakers)

Directory structure expected:
    MELD/
    +-- train/
    |   +-- train_sent_emo.csv
    |   +-- train_splits/
    +-- dev/
    |   +-- dev_sent_emo.csv
    |   +-- dev_splits_complete/
    +-- test/
        +-- test_sent_emo.csv
        +-- test_splits_complete/
"""

import csv
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict

from dataloader.base import (
    BaseEmotionDataset,
    CategoricalEmotion,
    DatasetSplit,
    EmotionSample,
    LABEL_MAPS,
)

logger = logging.getLogger(__name__)

SPLIT_DIRS = {
    DatasetSplit.TRAIN: "train",
    DatasetSplit.VALIDATION: "dev",
    DatasetSplit.TEST: "test",
}

CSV_PATTERNS = {
    DatasetSplit.TRAIN: ["train_sent_emo.csv", "train.csv"],
    DatasetSplit.VALIDATION: ["dev_sent_emo.csv", "dev.csv"],
    DatasetSplit.TEST: ["test_sent_emo.csv", "test.csv"],
}

AUDIO_DIR_PATTERNS = {
    DatasetSplit.TRAIN: ["train_splits", "train_audio", "audio"],
    DatasetSplit.VALIDATION: ["dev_splits_complete", "dev_splits", "dev_audio"],
    DatasetSplit.TEST: ["test_splits_complete", "test_splits", "test_audio"],
}


class MELDDataset(BaseEmotionDataset):
    """
    Loader for the MELD emotion dataset.

    Parameters
    ----------
    root_dir : str or Path
        Path to MELD/ directory.
    split : DatasetSplit
        Which split to load.
    context_turns : int
        Number of preceding turns to include as context (default: 3).
    emotions_filter : list of CategoricalEmotion, optional
        Only include these emotions.
    require_audio : bool
        Only include samples with existing audio files (default: False).
    max_samples : int, optional
        Cap number of samples.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: DatasetSplit = DatasetSplit.TRAIN,
        context_turns: int = 3,
        emotions_filter: Optional[list[CategoricalEmotion]] = None,
        require_audio: bool = False,
        max_samples: Optional[int] = None,
    ):
        super().__init__(root_dir, split, max_samples)
        self.context_turns = context_turns
        self.emotions_filter = emotions_filter
        self.require_audio = require_audio

    def _load_samples(self) -> list[EmotionSample]:
        csv_file = self._find_csv()
        if csv_file is None:
            logger.error(f"Cannot find MELD CSV for split={self.split.value}")
            return []

        audio_dir = self._find_audio_dir()
        dialogues = self._parse_csv(csv_file, audio_dir)
        samples = self._build_samples_with_context(dialogues)

        if self.emotions_filter:
            samples = [
                s for s in samples
                if s.categorical_emotion in self.emotions_filter
            ]

        logger.info(
            f"MELD: loaded {len(samples)} utterances from "
            f"{len(dialogues)} dialogues (split={self.split.value})"
        )
        return samples

    def _find_csv(self) -> Optional[Path]:
        """Locate the label CSV file for this split."""
        split_dir = self.root_dir / SPLIT_DIRS[self.split]
        patterns = CSV_PATTERNS[self.split]

        for pattern in patterns:
            path = split_dir / pattern
            if path.exists():
                return path

        for pattern in patterns:
            path = self.root_dir / pattern
            if path.exists():
                return path

        return None

    def _find_audio_dir(self) -> Optional[Path]:
        """Locate the audio directory for this split."""
        split_dir = self.root_dir / SPLIT_DIRS[self.split]
        patterns = AUDIO_DIR_PATTERNS[self.split]

        for pattern in patterns:
            path = split_dir / pattern
            if path.exists() and path.is_dir():
                return path

        if split_dir.exists() and list(split_dir.glob("*.wav")):
            return split_dir

        return None

    def _parse_csv(
        self, csv_file: Path, audio_dir: Optional[Path]
    ) -> dict[int, list[dict]]:
        """Parse MELD CSV and group utterances by dialogue."""
        dialogues: dict[int, list[dict]] = defaultdict(list)

        with open(csv_file, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    dialogue_id = int(row.get("Dialogue_ID", -1))
                    utterance_id = int(row.get("Utterance_ID", -1))
                except (ValueError, TypeError):
                    continue

                transcript = row.get("Utterance", "").strip()
                speaker = row.get("Speaker", "").strip()
                emotion_raw = row.get("Emotion", "neutral").strip().lower()
                sentiment = row.get("Sentiment", "").strip().lower()

                emotion = LABEL_MAPS["meld"].get(
                    emotion_raw, CategoricalEmotion.OTHER
                )

                # audio_path = ""
                # if audio_dir:
                #     wav_name = f"dia{dialogue_id}_utt{utterance_id}.wav"
                #     wav_path = audio_dir / wav_name
                #     if wav_path.exists():
                #         audio_path = str(wav_path)

                for ext in [".wav", ".mp4"]:
                    file_name = f"dia{dialogue_id}_utt{utterance_id}{ext}"
                    file_path = audio_dir / file_name
                    if file_path.exists():
                        audio_path = str(file_path)
                        break

                entry = {
                    "dialogue_id": dialogue_id,
                    "utterance_id": utterance_id,
                    "transcript": transcript,
                    "speaker": speaker,
                    "emotion": emotion,
                    "emotion_raw": emotion_raw,
                    "sentiment": sentiment,
                    "audio_path": audio_path,
                    "season": row.get("Season", ""),
                    "episode": row.get("Episode", ""),
                    "start_time": row.get("StartTime", ""),
                    "end_time": row.get("EndTime", ""),
                }
                dialogues[dialogue_id].append(entry)

        for dia_id in dialogues:
            dialogues[dia_id].sort(key=lambda x: x["utterance_id"])

        return dict(dialogues)

    def _build_samples_with_context(
        self, dialogues: dict[int, list[dict]]
    ) -> list[EmotionSample]:
        """Build EmotionSample objects with multi-party conversational context."""
        samples = []

        for dia_id, utterances in dialogues.items():
            for i, utt in enumerate(utterances):
                if self.require_audio and not utt["audio_path"]:
                    continue

                # Build context turns
                context = []
                start_idx = max(0, i - self.context_turns)
                for j in range(start_idx, i):
                    prev = utterances[j]
                    context.append(
                        f"[{prev['speaker']}] {prev['transcript']}"
                    )

                utt_id = f"meld_dia{dia_id}_utt{utt['utterance_id']}"

                sample = EmotionSample(
                    utterance_id=utt_id,
                    dataset_name="meld",
                    split=self.split,
                    audio_path=utt["audio_path"],
                    speaker_id=utt["speaker"],
                    transcript=utt["transcript"],
                    context_turns=context,
                    categorical_emotion=utt["emotion"],
                    emotion_raw_label=utt["emotion_raw"],
                    sentiment=utt["sentiment"],
                    metadata={
                        "dialogue_id": dia_id,
                        "utterance_idx": utt["utterance_id"],
                        "season": utt["season"],
                        "episode": utt["episode"],
                        "start_time": utt["start_time"],
                        "end_time": utt["end_time"],
                        "n_speakers_in_dialogue": len(set(
                            u["speaker"] for u in utterances
                        )),
                        "dialogue_length": len(utterances),
                    },
                )
                samples.append(sample)

        return samples

    @property
    def dialogue_ids(self) -> list[int]:
        """Unique dialogue IDs in this split."""
        return sorted(set(
            s.metadata.get("dialogue_id", -1) for s in self._samples
        ))

    def get_dialogue(self, dialogue_id: int) -> list[EmotionSample]:
        """Get all utterances from a specific dialogue, in order."""
        utterances = [
            s for s in self._samples
            if s.metadata.get("dialogue_id") == dialogue_id
        ]
        utterances.sort(key=lambda s: s.metadata.get("utterance_idx", 0))
        return utterances

    def get_multi_party_dialogues(
        self, min_speakers: int = 3
    ) -> list[list[EmotionSample]]:
        """
        Get dialogues with at least min_speakers unique speakers.
        Multi-party dialogues are the key OOD challenge vs dyadic IEMOCAP.
        """
        result = []
        for dia_id in self.dialogue_ids:
            dialogue = self.get_dialogue(dia_id)
            n_speakers = len(set(s.speaker_id for s in dialogue))
            if n_speakers >= min_speakers:
                result.append(dialogue)
        return result