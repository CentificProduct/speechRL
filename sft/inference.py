"""
Emotion-GSRM: Inference
=========================
Inference pipeline for the fine-tuned Qwen2.5-Omni-7B model.

From the proposal:
  - K=16 samples averaged at temperature 1.0, top-p 0.6
  - Input: raw audio
  - Output: evidence log + CoT reasoning + 7-dimension scores

The K-sample averaging strategy follows GSRM: generate K independent
CoT samples, parse scores from each, and average to reduce variance.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import numpy as np

from emotion_gsrm.rubric import DimensionName, RubricPromptBuilder, EmotionScores
from emotion_gsrm.training.dataset import (
    SYSTEM_PROMPT, INPUT_TEMPLATE, CONTEXT_TEMPLATE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference Result
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Result from a single inference run (K samples averaged)."""
    utterance_id: str
    scores: dict[str, float]
    score_stds: dict[str, float]
    reasoning: str
    evidence: str
    k_samples: int
    k_scores: list[dict[str, float]]
    k_reasonings: list[str]
    elapsed_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_emotion_scores(self) -> EmotionScores:
        """Convert averaged scores to EmotionScores object."""
        return EmotionScores.from_dict(
            self.scores, utterance_id=self.utterance_id,
        )

    def to_dict(self) -> dict:
        return {
            "utterance_id": self.utterance_id,
            "scores": self.scores,
            "score_stds": self.score_stds,
            "reasoning": self.reasoning,
            "evidence": self.evidence,
            "k_samples": self.k_samples,
            "k_scores": self.k_scores,
            "elapsed_seconds": self.elapsed_seconds,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------

class EmotionGSRMInference:
    """
    Inference engine for the fine-tuned Emotion-GSRM model.

    Implements K-sample averaging: generates K independent chain-of-thought
    responses at temperature 1.0, parses scores from each, and averages
    to produce final predictions.

    Parameters
    ----------
    model_path : str or Path
        Path to fine-tuned model checkpoint.
    k : int
        Number of samples to average (default: 16).
    temperature : float
        Sampling temperature (default: 1.0).
    top_p : float
        Nucleus sampling parameter (default: 0.6).
    max_new_tokens : int
        Maximum generation length (default: 2048).
    device : str
        Device for inference ('cuda', 'cpu', 'auto').
    torch_dtype : str
        Model precision ('bfloat16', 'float16', 'float32').
    batch_k : bool
        Batch K samples in a single forward pass (default: True).
    """

    def __init__(
        self,
        model_path: str | Path,
        k: int = 16,
        temperature: float = 1.0,
        top_p: float = 0.6,
        max_new_tokens: int = 2048,
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        batch_k: bool = True,
    ):
        self.model_path = Path(model_path)
        self.k = k
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.torch_dtype = torch_dtype
        self.batch_k = batch_k

        self._model = None
        self._tokenizer = None
        self._processor = None

    def load_model(self) -> "EmotionGSRMInference":
        """Load the fine-tuned model and tokenizer."""
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                AutoProcessor,
            )
        except ImportError:
            raise ImportError(
                "transformers required: pip install transformers"
            )

        logger.info(f"Loading model from {self.model_path}")
        dtype = getattr(torch, self.torch_dtype, torch.bfloat16)

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=self.device,
        )
        self._model.eval()

        try:
            self._processor = AutoProcessor.from_pretrained(
                str(self.model_path), trust_remote_code=True,
            )
        except Exception:
            self._processor = None

        logger.info("Model loaded successfully")
        return self

    def predict(
        self,
        audio_path: str,
        transcript: str = "",
        context_turns: Optional[list[str]] = None,
        utterance_id: str = "",
    ) -> InferenceResult:
        """
        Run K-sample inference on a single utterance.

        Parameters
        ----------
        audio_path : str
            Path to audio file.
        transcript : str
            Utterance transcript.
        context_turns : list of str, optional
            Preceding conversational turns.
        utterance_id : str
            Identifier for this utterance.

        Returns
        -------
        InferenceResult
            Averaged scores with per-sample details.
        """
        if self._model is None:
            raise RuntimeError("Call load_model() first.")

        start_time = time.time()
        prompt = self._build_prompt(transcript, context_turns)
        k_responses = self._generate_k_samples(prompt, audio_path)

        k_scores = []
        k_reasonings = []
        k_evidences = []

        for response in k_responses:
            scores = RubricPromptBuilder.parse_scores_from_response(response)
            reasoning = self._extract_reasoning(response)
            evidence = self._extract_evidence(response)
            k_scores.append(scores)
            k_reasonings.append(reasoning)
            k_evidences.append(evidence)

        avg_scores, score_stds = self._average_scores(k_scores)
        best_idx = self._select_best_sample(k_scores, avg_scores)
        elapsed = time.time() - start_time

        return InferenceResult(
            utterance_id=utterance_id,
            scores=avg_scores,
            score_stds=score_stds,
            reasoning=k_reasonings[best_idx] if k_reasonings else "",
            evidence=k_evidences[best_idx] if k_evidences else "",
            k_samples=len(k_responses),
            k_scores=k_scores,
            k_reasonings=k_reasonings,
            elapsed_seconds=round(elapsed, 2),
            metadata={
                "k": self.k,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "audio_path": audio_path,
                "best_sample_idx": best_idx,
            },
        )

    def predict_batch(
        self, samples: list[dict],
    ) -> list[InferenceResult]:
        """Run inference on a batch of utterances."""
        results = []
        for i, sample in enumerate(samples):
            logger.info(
                f"Predicting {i+1}/{len(samples)}: "
                f"{sample.get('utterance_id', '')}"
            )
            result = self.predict(
                audio_path=sample["audio_path"],
                transcript=sample.get("transcript", ""),
                context_turns=sample.get("context_turns"),
                utterance_id=sample.get("utterance_id", f"sample_{i}"),
            )
            results.append(result)
        return results

    def _build_prompt(
        self, transcript: str, context_turns: Optional[list[str]] = None,
    ) -> str:
        """Build the input prompt matching training format."""
        context_block = ""
        if context_turns:
            context_text = "\n".join(f"  {t}" for t in context_turns)
            context_block = CONTEXT_TEMPLATE.format(context=context_text)

        user_content = INPUT_TEMPLATE.format(
            context_block=context_block, transcript=transcript,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        if self._processor and hasattr(self._processor, "apply_chat_template"):
            return self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
        elif self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
        else:
            return (
                f"<|system|>\n{SYSTEM_PROMPT}\n"
                f"<|user|>\n{user_content}\n"
                f"<|assistant|>\n"
            )

    def _generate_k_samples(self, prompt: str, audio_path: str) -> list[str]:
        """Generate K independent samples from the model."""
        import torch

        inputs = self._tokenizer(
            prompt, return_tensors="pt",
            max_length=self.max_new_tokens, truncation=True,
        ).to(self._model.device)

        responses = []

        if self.batch_k:
            batch_input_ids = inputs["input_ids"].repeat(self.k, 1)
            batch_attention = inputs["attention_mask"].repeat(self.k, 1)

            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            for i in range(self.k):
                generated = outputs[i, input_len:]
                text = self._tokenizer.decode(
                    generated, skip_special_tokens=True
                )
                responses.append(text)
        else:
            for _ in range(self.k):
                with torch.no_grad():
                    output = self._model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        pad_token_id=self._tokenizer.pad_token_id,
                    )
                input_len = inputs["input_ids"].shape[1]
                generated = output[0, input_len:]
                text = self._tokenizer.decode(
                    generated, skip_special_tokens=True
                )
                responses.append(text)

        return responses

    def _average_scores(
        self, k_scores: list[dict[str, float]]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Average scores across K samples and compute std."""
        if not k_scores:
            default = {d.value: 3.0 for d in DimensionName}
            return default, {d.value: 0.0 for d in DimensionName}

        avg_scores = {}
        score_stds = {}
        for dim in DimensionName:
            values = [s.get(dim.value, 3.0) for s in k_scores]
            avg_scores[dim.value] = round(float(np.mean(values)), 2)
            score_stds[dim.value] = round(float(np.std(values)), 3)

        return avg_scores, score_stds

    def _select_best_sample(
        self, k_scores: list[dict[str, float]], avg_scores: dict[str, float],
    ) -> int:
        """Select the sample closest to the average scores."""
        if not k_scores:
            return 0
        min_dist = float("inf")
        best_idx = 0
        for i, scores in enumerate(k_scores):
            dist = sum(
                (scores.get(d.value, 3.0) - avg_scores.get(d.value, 3.0)) ** 2
                for d in DimensionName
            )
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        return best_idx

    @staticmethod
    def _extract_reasoning(response: str) -> str:
        if "[REASONING]" in response:
            parts = response.split("[REASONING]", 1)
            remainder = parts[1] if len(parts) > 1 else ""
            if "[SCORES]" in remainder:
                return remainder.split("[SCORES]")[0].strip()
            return remainder.strip()
        return ""

    @staticmethod
    def _extract_evidence(response: str) -> str:
        if "[EVIDENCE]" in response:
            parts = response.split("[EVIDENCE]", 1)
            remainder = parts[1] if len(parts) > 1 else ""
            if "[REASONING]" in remainder:
                return remainder.split("[REASONING]")[0].strip()
            return remainder.strip()
        return ""

    def export_results(
        self, results: list[InferenceResult], output_path: str | Path,
    ) -> None:
        """Export inference results to JSONL."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for result in results:
                f.write(json.dumps(result.to_dict()) + "\n")
        logger.info(f"Exported {len(results)} results to {output_path}")