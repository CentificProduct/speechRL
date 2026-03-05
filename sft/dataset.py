"""
Emotion-GSRM: SFT Dataset & Collator
======================================
Converts CoT synthesis output into training-ready format for
Qwen2.5-Omni-7B fine-tuning.

Input format: raw audio -> evidence log + CoT + scores
The model learns to generate the full chain from audio input.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert speech emotion evaluator. Given a speech utterance, "
    "analyze its emotional expressivity by examining acoustic features, "
    "then provide a detailed chain-of-thought assessment and scores "
    "across 7 emotion dimensions on a 1-5 scale."
)

INPUT_TEMPLATE = (
    "<|audio|>\n"
    "Evaluate the emotional expressivity of this speech utterance.\n"
    "{context_block}"
    "Transcript: \"{transcript}\"\n\n"
    "Provide:\n"
    "1. [EVIDENCE] - Acoustic feature analysis per dimension\n"
    "2. [REASONING] - Chain-of-thought connecting evidence to scores\n"
    "3. [SCORES] - Ratings for all 7 emotion dimensions (1-5)"
)

CONTEXT_TEMPLATE = (
    "Conversational context (preceding turns):\n{context}\n\n"
)


# ---------------------------------------------------------------------------
# SFT Dataset
# ---------------------------------------------------------------------------

class SFTDataset:
    """
    Dataset for supervised fine-tuning of Qwen2.5-Omni-7B.

    Loads CoT synthesis JSONL and formats samples as input-output
    pairs suitable for the SWIFT training framework.

    Parameters
    ----------
    data_path : str or Path
        Path to sft_training_data.jsonl from CoT synthesis.
    max_length : int
        Maximum token length (default: 4096).
    audio_dir : str or Path, optional
        Override audio directory (if audio paths in JSONL are relative).
    val_split : float
        Fraction to reserve for validation (default: 0.0 = no split).
    seed : int
        Random seed for splitting (default: 42).
    """

    def __init__(
        self,
        data_path: str | Path,
        max_length: int = 4096,
        audio_dir: Optional[str | Path] = None,
        val_split: float = 0.0,
        seed: int = 42,
    ):
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.val_split = val_split
        self.seed = seed

        self._samples: list[dict] = []
        self._val_samples: list[dict] = []
        self._loaded = False

    def load(self) -> "SFTDataset":
        """Load and format training samples from JSONL."""
        if self._loaded:
            return self

        raw_samples = []
        with open(self.data_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    formatted = self._format_sample(data)
                    if formatted:
                        raw_samples.append(formatted)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping line {line_num}: {e}")

        # Split if needed
        if self.val_split > 0 and len(raw_samples) > 10:
            random.seed(self.seed)
            random.shuffle(raw_samples)
            n_val = int(len(raw_samples) * self.val_split)
            self._val_samples = raw_samples[:n_val]
            self._samples = raw_samples[n_val:]
        else:
            self._samples = raw_samples

        self._loaded = True
        logger.info(
            f"Loaded {len(self._samples)} train, "
            f"{len(self._val_samples)} val samples"
        )
        return self

    @property
    def train_samples(self) -> list[dict]:
        return self._samples

    @property
    def val_samples(self) -> list[dict]:
        return self._val_samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]

    def _format_sample(self, data: dict) -> Optional[dict]:
        """
        Format a single CoT sample into SWIFT-compatible training format.

        SWIFT expects:
          - "messages": list of role/content dicts (chat format)
          - "audios": list of audio file paths
        """
        transcript = data.get("transcript", "")
        audio_path = data.get("audio_path", "")
        context_turns = data.get("context_turns", [])
        target_output = data.get("target_output", "")

        if not target_output:
            return None

        # Resolve audio path
        if self.audio_dir and audio_path and not Path(audio_path).is_absolute():
            audio_path = str(self.audio_dir / audio_path)

        # Build context block
        context_block = ""
        if context_turns:
            context_text = "\n".join(f"  {t}" for t in context_turns)
            context_block = CONTEXT_TEMPLATE.format(context=context_text)

        # Build user input
        user_content = INPUT_TEMPLATE.format(
            context_block=context_block,
            transcript=transcript,
        )

        # SWIFT chat format
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target_output},
        ]

        return {
            "messages": messages,
            "audios": [audio_path] if audio_path else [],
            "utterance_id": data.get("utterance_id", ""),
            "scores": data.get("scores", {}),
        }

    def to_swift_jsonl(self, output_path: str | Path) -> None:
        """
        Export in SWIFT-native JSONL format.

        This is the format directly consumed by `swift sft`.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for sample in self._samples:
                swift_entry = {
                    "messages": sample["messages"],
                }
                if sample.get("audios"):
                    swift_entry["audios"] = sample["audios"]
                f.write(json.dumps(swift_entry) + "\n")

        logger.info(f"Exported {len(self._samples)} samples to {output_path}")

        # Export val set if present
        if self._val_samples:
            val_path = output_path.parent / f"val_{output_path.name}"
            with open(val_path, "w") as f:
                for sample in self._val_samples:
                    swift_entry = {"messages": sample["messages"]}
                    if sample.get("audios"):
                        swift_entry["audios"] = sample["audios"]
                    f.write(json.dumps(swift_entry) + "\n")
            logger.info(
                f"Exported {len(self._val_samples)} val samples to {val_path}"
            )

    def to_hf_dataset(self):
        """
        Convert to a HuggingFace Dataset object.

        Alternative path for users preferring the HF training ecosystem.
        Requires: pip install datasets
        """
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError("datasets package required: pip install datasets")

        records = []
        for sample in self._samples:
            msgs = sample["messages"]
            records.append({
                "system": msgs[0]["content"] if msgs else "",
                "input": msgs[1]["content"] if len(msgs) > 1 else "",
                "output": msgs[2]["content"] if len(msgs) > 2 else "",
                "audio_path": sample.get("audios", [""])[0],
                "utterance_id": sample.get("utterance_id", ""),
            })

        return Dataset.from_list(records)

    def summary(self) -> str:
        """Print dataset summary statistics."""
        if not self._samples:
            return "Dataset not loaded"

        output_lengths = []
        has_audio = 0
        has_context = 0

        for s in self._samples:
            msgs = s["messages"]
            if len(msgs) > 2:
                output_lengths.append(len(msgs[2]["content"]))
            if s.get("audios"):
                has_audio += 1
            if any("context" in msgs[1]["content"].lower() for _ in [1]):
                has_context += 1

        import numpy as np
        return (
            f"SFT Dataset Summary:\n"
            f"  Train samples: {len(self._samples)}\n"
            f"  Val samples: {len(self._val_samples)}\n"
            f"  With audio: {has_audio}\n"
            f"  With context: {has_context}\n"
            f"  Output length (chars): "
            f"mean={np.mean(output_lengths):.0f}, "
            f"max={np.max(output_lengths):.0f}, "
            f"min={np.min(output_lengths):.0f}"
        )


# ---------------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------------

@dataclass
class SFTCollator:
    """
    Custom data collator for Emotion-GSRM SFT.

    Handles batching of text + audio multimodal inputs for
    Qwen2.5-Omni-7B. Pads sequences and loads audio on-the-fly.

    Parameters
    ----------
    tokenizer : Any
        The model tokenizer.
    processor : Any
        The model processor (handles audio encoding).
    max_length : int
        Maximum sequence length.
    padding : str
        Padding strategy ('max_length' or 'longest').
    """
    tokenizer: Any = None
    processor: Any = None
    max_length: int = 4096
    padding: str = "longest"

    def __call__(self, features: list[dict]) -> dict:
        """
        Collate a batch of samples.

        Each feature dict contains 'messages' and optionally 'audios'.
        """
        batch_messages = []
        batch_audios = []

        for feature in features:
            messages = feature.get("messages", [])
            audios = feature.get("audios", [])

            batch_messages.append(messages)
            batch_audios.append(audios)

        # Process with model-specific processor
        if self.processor is not None:
            return self._process_with_qwen(batch_messages, batch_audios)
        elif self.tokenizer is not None:
            return self._process_text_only(batch_messages)
        else:
            raise ValueError("Either tokenizer or processor must be provided")

    def _process_with_qwen(
        self, batch_messages: list, batch_audios: list
    ) -> dict:
        """
        Process batch using Qwen2.5-Omni processor.

        Handles multimodal (text + audio) input encoding.
        """
        processed_inputs = []
        processed_labels = []

        for messages, audios in zip(batch_messages, batch_audios):
            # Separate input (system + user) from target (assistant)
            input_messages = messages[:2]  # system + user
            target_text = messages[2]["content"] if len(messages) > 2 else ""

            # Apply chat template for input
            input_text = self.processor.apply_chat_template(
                input_messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            # Full text for labels
            full_text = input_text + target_text

            # Tokenize
            input_ids = self.tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )

            # Create labels (mask input portion, keep target)
            input_only = self.tokenizer(
                input_text,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_len = input_only["input_ids"].shape[1]

            labels = input_ids["input_ids"].clone()
            labels[0, :input_len] = -100  # Mask input tokens

            processed_inputs.append(input_ids)
            processed_labels.append(labels)

        # Pad batch
        return self._pad_batch(processed_inputs, processed_labels)

    def _process_text_only(self, batch_messages: list) -> dict:
        """Fallback text-only processing (no audio)."""
        texts = []
        for messages in batch_messages:
            # Concatenate all message contents
            full_text = "\n".join(
                f"<|{m['role']}|>\n{m['content']}" for m in messages
            )
            texts.append(full_text)

        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=self.padding,
            return_tensors="pt",
        )

        # Labels = input_ids for causal LM
        encoded["labels"] = encoded["input_ids"].clone()

        return encoded

    def _pad_batch(self, inputs: list, labels: list) -> dict:
        """Pad a batch of variable-length sequences."""
        import torch

        max_len = max(inp["input_ids"].shape[1] for inp in inputs)
        if self.padding == "max_length":
            max_len = self.max_length

        pad_id = self.tokenizer.pad_token_id or 0

        batch_input_ids = []
        batch_attention = []
        batch_labels = []

        for inp, lbl in zip(inputs, labels):
            seq_len = inp["input_ids"].shape[1]
            pad_len = max_len - seq_len

            if pad_len > 0:
                padding = torch.full((1, pad_len), pad_id, dtype=torch.long)
                label_padding = torch.full((1, pad_len), -100, dtype=torch.long)
                attn_padding = torch.zeros((1, pad_len), dtype=torch.long)

                batch_input_ids.append(
                    torch.cat([inp["input_ids"], padding], dim=1)
                )
                batch_attention.append(
                    torch.cat([inp["attention_mask"], attn_padding], dim=1)
                )
                batch_labels.append(
                    torch.cat([lbl, label_padding], dim=1)
                )
            else:
                batch_input_ids.append(inp["input_ids"][:, :max_len])
                batch_attention.append(inp["attention_mask"][:, :max_len])
                batch_labels.append(lbl[:, :max_len])

        return {
            "input_ids": torch.cat(batch_input_ids, dim=0),
            "attention_mask": torch.cat(batch_attention, dim=0),
            "labels": torch.cat(batch_labels, dim=0),
        }