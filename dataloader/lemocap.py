"""
IEMOCAP Dataset Loader
=======================
Interactive Emotional Dyadic Motion Capture database.

- Size: ~12 hours, ~10K utterances
- Labels: Categorical emotion + VAD dimensional (1-5 scale)
- Role: Primary training dataset for CoT synthesis
- Structure: 5 sessions, each with 2 speakers in scripted + improvised dialogues

Directory structure expected:
    IEMOCAP_full_release/
    +-- Session1/
    |   +-- dialog/
    |   |   +-- EmoEvaluation/       # Emotion annotations
    |   |   +-- transcriptions/      # Utterance transcripts
    |   +-- sentences/
    |       +-- wav/                  # Individual utterance wavs
    +-- Session2/
    +-- ...
"""

import logging
import re
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

# Standard IEMOCAP leave-one-session-out splits
DEFAULT_SPLITS = {
    DatasetSplit.TRAIN: [1, 2, 3, 4],
    DatasetSplit.TEST: [5],
    DatasetSplit.VALIDATION: [5],
}

# Regex for parsing IEMOCAP evaluation files
# Format: [start - end] utterance_id emotion [valence, arousal, dominance]
EVAL_LINE_RE = re.compile(
    r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+"
    r"(\S+)\s+"
    r"(\w+)\s+"
    r"\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]"
)


class IEMOCAPDataset(BaseEmotionDataset):
    """
    Loader for the IEMOCAP emotion speech database.

    Parameters
    ----------
    root_dir : str or Path
        Path to IEMOCAP_full_release/ directory.
    split : DatasetSplit
        Which split to load.
    sessions : list of int, optional
        Override which sessions to include. If None, uses DEFAULT_SPLITS.
    include_scripted : bool
        Whether to include scripted dialogues (default: True).
    include_improvised : bool
        Whether to include improvised dialogues (default: True).
    emotions_filter : list of CategoricalEmotion, optional
        Only include these emotions. Common: [neutral, happy, sad, angry].
    merge_excited_happy : bool
        Merge 'excited' into 'happy' category (default: True, standard practice).
    max_samples : int, optional
        Cap number of samples.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: DatasetSplit = DatasetSplit.TRAIN,
        sessions: Optional[list[int]] = None,
        include_scripted: bool = True,
        include_improvised: bool = True,
        emotions_filter: Optional[list[CategoricalEmotion]] = None,
        merge_excited_happy: bool = True,
        max_samples: Optional[int] = None,
    ):
        super().__init__(root_dir, split, max_samples)
        self.sessions = sessions or DEFAULT_SPLITS.get(split, [1, 2, 3, 4])
        self.include_scripted = include_scripted
        self.include_improvised = include_improvised
        self.emotions_filter = emotions_filter
        self.merge_excited_happy = merge_excited_happy

    def _load_samples(self) -> list[EmotionSample]:
        samples = []

        for session_num in self.sessions:
            session_dir = self.root_dir / f"Session{session_num}"
            if not session_dir.exists():
                logger.warning(f"Session directory not found: {session_dir}")
                continue

            eval_dir = session_dir / "dialog" / "EmoEvaluation"
            trans_dir = session_dir / "dialog" / "transcriptions"
            wav_dir = session_dir / "sentences" / "wav"

            if not eval_dir.exists():
                logger.warning(f"Evaluation dir not found: {eval_dir}")
                continue

            # Load all transcripts for this session
            transcripts = self._load_transcripts(trans_dir)

            # Parse evaluation files
            for eval_file in sorted(eval_dir.glob("*.txt")):
                dialog_id = eval_file.stem

                # Filter by dialog type
                is_improvised = "impro" in dialog_id
                is_scripted = "script" in dialog_id
                if is_improvised and not self.include_improvised:
                    continue
                if is_scripted and not self.include_scripted:
                    continue

                # Parse annotations
                dialog_samples = self._parse_evaluation_file(
                    eval_file, wav_dir, transcripts, session_num
                )

                # Build context turns for each sample within dialog
                dialog_samples = self._attach_context(dialog_samples)
                samples.extend(dialog_samples)

        # Apply emotion filter
        if self.emotions_filter:
            samples = [
                s for s in samples
                if s.categorical_emotion in self.emotions_filter
            ]

        logger.info(
            f"IEMOCAP: loaded {len(samples)} utterances from "
            f"sessions {self.sessions}"
        )
        return samples

    def _load_transcripts(self, trans_dir: Path) -> dict[str, str]:
        """Load all transcripts from a session's transcription directory."""
        transcripts = {}
        if not trans_dir.exists():
            return transcripts

        for trans_file in trans_dir.glob("*.txt"):
            with open(trans_file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(";"):
                        continue
                    match = re.match(
                        r"(\S+)\s+\[\d+\.\d+-\d+\.\d+\]:\s*(.*)", line
                    )
                    if match:
                        utt_id = match.group(1)
                        text = match.group(2).strip()
                        transcripts[utt_id] = text

        return transcripts

    def _parse_evaluation_file(
        self,
        eval_file: Path,
        wav_dir: Path,
        transcripts: dict[str, str],
        session_num: int,
    ) -> list[EmotionSample]:
        """Parse a single IEMOCAP evaluation file."""
        samples = []

        with open(eval_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                match = EVAL_LINE_RE.match(line.strip())
                if not match:
                    continue

                start_time = float(match.group(1))
                end_time = float(match.group(2))
                utt_id = match.group(3)
                emotion_raw = match.group(4).lower()
                valence = float(match.group(5))
                arousal = float(match.group(6))
                dominance = float(match.group(7))

                # Map emotion label
                emotion = LABEL_MAPS["iemocap"].get(
                    emotion_raw[:3], CategoricalEmotion.OTHER
                )

                # Merge excited -> happy if configured
                if self.merge_excited_happy and emotion == CategoricalEmotion.EXCITED:
                    emotion = CategoricalEmotion.HAPPY

                # Resolve audio path
                dialog_id = "_".join(utt_id.split("_")[:2])
                audio_path = wav_dir / dialog_id / f"{utt_id}.wav"

                # Determine speaker from utterance ID
                speaker_gender = "F" if "_F" in utt_id.split("_")[-1][:2] else "M"
                speaker_id = f"S{session_num:02d}_{speaker_gender}"

                sample = EmotionSample(
                    utterance_id=utt_id,
                    dataset_name="iemocap",
                    split=self.split,
                    audio_path=str(audio_path),
                    duration_sec=end_time - start_time,
                    speaker_id=speaker_id,
                    gender=speaker_gender,
                    transcript=transcripts.get(utt_id, ""),
                    categorical_emotion=emotion,
                    emotion_raw_label=emotion_raw,
                    vad_annotations=[
                        VADAnnotation(
                            valence=valence,
                            arousal=arousal,
                            dominance=dominance,
                            annotator_id="majority_vote",
                        )
                    ],
                    metadata={
                        "session": session_num,
                        "dialog_id": dialog_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "dialog_type": (
                            "improvised" if "impro" in dialog_id else "scripted"
                        ),
                    },
                )
                samples.append(sample)

        return samples

    def _attach_context(
        self, dialog_samples: list[EmotionSample], n_turns: int = 3
    ) -> list[EmotionSample]:
        """
        Attach preceding conversational turns to each sample.
        IEMOCAP dialogues are dyadic, so we use preceding utterances
        (from both speakers) as context.
        """
        dialog_samples.sort(key=lambda s: s.metadata.get("start_time", 0))

        for i, sample in enumerate(dialog_samples):
            context = []
            for j in range(max(0, i - n_turns), i):
                prev = dialog_samples[j]
                speaker_tag = f"[{prev.speaker_id}]"
                context.append(f"{speaker_tag} {prev.transcript}")
            sample.context_turns = context

        return dialog_samples