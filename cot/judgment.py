"""
Emotion-GSRM: Stage 2 — Global Emotion Judgment CoT Synthesis
===============================================================
GPT-4o receives the full evidence log plus oracle ratings and synthesizes
a coherent chain-of-thought assessment connecting acoustic evidence to
emotion scores. The model must match oracle ratings while making reasoning
appear self-derived.

Output format:
  [REASONING]
  <chain-of-thought analysis connecting evidence to scores>
  [SCORES]
  emotional_intensity: <1-5>
  emotional_appropriateness: <1-5>
  ...
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from cot.evidence import EvidenceLog
from rubric.init import (
    DimensionName,
    RubricPromptBuilder,
    EmotionScores,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CoT Sample Data Structure
# ---------------------------------------------------------------------------

@dataclass
class CoTSample:
    """
    A complete CoT training sample for SFT.

    Contains the evidence log, the chain-of-thought reasoning,
    and the final scores — everything needed to train the student
    model (Qwen2.5-Omni-7B).
    """
    utterance_id: str
    evidence_log: EvidenceLog
    reasoning: str
    scores: dict[str, float]
    oracle_scores: Optional[dict[str, float]] = None
    generation_metadata: dict = field(default_factory=dict)

    def to_training_format(self) -> dict:
        """
        Format as a training sample for SFT.

        Returns the input-output pair expected by the SWIFT training
        framework: raw audio path + expected model output.
        """
        # Build the expected model output (evidence + CoT + scores)
        output_text = self._build_target_output()

        return {
            "utterance_id": self.utterance_id,
            "audio_path": self.evidence_log.generation_metadata.get(
                "audio_path", ""
            ),
            "transcript": self.evidence_log.transcript,
            "context_turns": self.evidence_log.context_turns,
            "target_output": output_text,
            "scores": self.scores,
            "oracle_scores": self.oracle_scores,
        }

    def _build_target_output(self) -> str:
        """Build the full target output string for SFT training."""
        lines = [
            "[EVIDENCE]",
            self.evidence_log.to_prompt_text(),
            "",
            "[REASONING]",
            self.reasoning,
            "",
            "[SCORES]",
        ]
        for dim in DimensionName:
            score = self.scores.get(dim.value, 3.0)
            lines.append(f"{dim.value}: {score}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        return {
            "utterance_id": self.utterance_id,
            "evidence_log": self.evidence_log.to_dict(),
            "reasoning": self.reasoning,
            "scores": self.scores,
            "oracle_scores": self.oracle_scores,
            "generation_metadata": self.generation_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CoTSample":
        """Deserialize from JSON."""
        return cls(
            utterance_id=data["utterance_id"],
            evidence_log=EvidenceLog.from_dict(data["evidence_log"]),
            reasoning=data["reasoning"],
            scores=data["scores"],
            oracle_scores=data.get("oracle_scores"),
            generation_metadata=data.get("generation_metadata", {}),
        )


# ---------------------------------------------------------------------------
# Stage 2 System Prompt
# ---------------------------------------------------------------------------

STAGE2_SYSTEM_PROMPT = """\
You are an expert speech emotion evaluator synthesizing a global judgment \
from detailed acoustic evidence. You will receive:

1. A complete evidence log with per-dimension acoustic analysis
2. Oracle ratings that your reasoning must arrive at naturally

Your task:
- Write a coherent chain-of-thought that connects specific acoustic \
evidence to emotion scores across all 7 dimensions
- Identify cross-dimensional patterns (e.g., high intensity but poor \
appropriateness)
- Your reasoning must lead naturally to the target scores WITHOUT \
appearing to work backward from them
- The reasoning should read as though you are deriving the scores from \
first principles based on the evidence

Output format:
[REASONING]
<Your detailed chain-of-thought analysis>
[SCORES]
emotional_intensity: <score>
emotional_appropriateness: <score>
emotional_consistency: <score>
valence_accuracy: <score>
arousal_accuracy: <score>
transition_smoothness: <score>
overall_emotional_quality: <score>

Rules:
- ALWAYS ground reasoning in specific acoustic features from the evidence
- Connect cross-dimensional observations (e.g., pitch variation relates \
to both intensity and consistency)
- Scores must match the oracle ratings provided
- Reasoning should be 150-300 words
- Make the reasoning flow naturally, as if discovering the scores
"""


# ---------------------------------------------------------------------------
# Judgment Synthesizer
# ---------------------------------------------------------------------------

class JudgmentSynthesizer:
    """
    Stage 2: Synthesize global emotion judgment CoT from evidence logs.

    Takes Stage 1 evidence logs + oracle scores and produces training
    samples with chain-of-thought reasoning that appears self-derived.

    Parameters
    ----------
    api_key : str
        OpenAI API key.
    model : str
        Teacher model name (default: "gpt-4o").
    temperature : float
        Sampling temperature (default: 0.4).
    max_retries : int
        Max API call retries (default: 3).
    retry_delay : float
        Base delay between retries in seconds (default: 1.0).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.4,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._client = None
        self._call_count = 0

    @property
    def client(self):
        """Lazy-initialize OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                )
        return self._client

    def synthesize(
        self,
        evidence_log: EvidenceLog,
        oracle_scores: dict[str, float],
    ) -> CoTSample:
        """
        Synthesize a CoT judgment for one utterance.

        Parameters
        ----------
        evidence_log : EvidenceLog
            Stage 1 output with per-dimension evidence.
        oracle_scores : dict
            Target scores (from human annotations) that the CoT
            must arrive at naturally.

        Returns
        -------
        CoTSample
            Complete training sample with evidence + reasoning + scores.
        """
        start_time = time.time()

        prompt = RubricPromptBuilder.build_judgment_prompt(
            evidence_log=evidence_log.to_prompt_text(),
            oracle_scores=oracle_scores,
        )

        response = self._call_teacher(prompt)
        reasoning, scores = self._parse_judgment_response(response, oracle_scores)

        elapsed = time.time() - start_time

        return CoTSample(
            utterance_id=evidence_log.utterance_id,
            evidence_log=evidence_log,
            reasoning=reasoning,
            scores=scores,
            oracle_scores=oracle_scores,
            generation_metadata={
                "model": self.model,
                "temperature": self.temperature,
                "elapsed_seconds": round(elapsed, 2),
                "score_match": self._check_score_match(scores, oracle_scores),
            },
        )

    def synthesize_with_verification(
        self,
        evidence_log: EvidenceLog,
        oracle_scores: dict[str, float],
        max_attempts: int = 3,
        tolerance: float = 0.5,
    ) -> CoTSample:
        """
        Synthesize with verification that scores match oracle.

        Retries if the model's self-derived scores deviate too far
        from oracle scores, requesting it to adjust reasoning.

        Parameters
        ----------
        evidence_log : EvidenceLog
            Stage 1 evidence log.
        oracle_scores : dict
            Target scores.
        max_attempts : int
            Maximum synthesis attempts (default: 3).
        tolerance : float
            Max allowed deviation per dimension (default: 0.5).
        """
        for attempt in range(max_attempts):
            sample = self.synthesize(evidence_log, oracle_scores)

            # Check if scores match within tolerance
            deviations = {
                dim: abs(sample.scores.get(dim, 3.0) - oracle_scores.get(dim, 3.0))
                for dim in oracle_scores
            }
            max_dev = max(deviations.values()) if deviations else 0.0

            if max_dev <= tolerance:
                logger.debug(
                    f"Score match achieved (max deviation: {max_dev:.2f}) "
                    f"on attempt {attempt + 1}"
                )
                return sample

            logger.debug(
                f"Score mismatch (max deviation: {max_dev:.2f}), "
                f"retrying ({attempt + 1}/{max_attempts})"
            )

        logger.warning(
            f"Could not achieve exact score match for "
            f"{evidence_log.utterance_id} after {max_attempts} attempts. "
            f"Using oracle scores with last reasoning."
        )
        # Force oracle scores with last reasoning
        sample.scores = oracle_scores.copy()
        return sample

    def _call_teacher(self, user_prompt: str) -> str:
        """Call GPT-4o with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": STAGE2_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=2000,
                )
                self._call_count += 1
                return response.choices[0].message.content or ""

            except Exception as e:
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/"
                    f"{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))

        return ""

    def _parse_judgment_response(
        self,
        response: str,
        oracle_scores: dict[str, float],
    ) -> tuple[str, dict[str, float]]:
        """
        Parse the [REASONING] and [SCORES] sections from the response.

        Falls back to oracle scores if parsing fails.
        """
        reasoning = ""
        scores = {}

        if "[REASONING]" in response:
            parts = response.split("[REASONING]", 1)
            remainder = parts[1] if len(parts) > 1 else ""

            if "[SCORES]" in remainder:
                reasoning_part, scores_part = remainder.split("[SCORES]", 1)
                reasoning = reasoning_part.strip()
                scores = RubricPromptBuilder.parse_scores_from_response(
                    "[SCORES]\n" + scores_part
                )
            else:
                reasoning = remainder.strip()
        elif "[SCORES]" in response:
            scores = RubricPromptBuilder.parse_scores_from_response(response)
            # Everything before [SCORES] is reasoning
            reasoning = response.split("[SCORES]")[0].strip()
        else:
            # Fallback: treat entire response as reasoning
            reasoning = response.strip()

        # Fill missing scores from oracle
        if not scores:
            scores = oracle_scores.copy()
        else:
            for dim in DimensionName:
                if dim.value not in scores:
                    scores[dim.value] = oracle_scores.get(dim.value, 3.0)

        return reasoning, scores

    @staticmethod
    def _check_score_match(
        generated: dict[str, float],
        oracle: dict[str, float],
        tolerance: float = 0.5,
    ) -> dict:
        """Check how well generated scores match oracle scores."""
        deviations = {}
        for dim in oracle:
            gen_val = generated.get(dim, 3.0)
            oracle_val = oracle[dim]
            deviations[dim] = {
                "generated": gen_val,
                "oracle": oracle_val,
                "deviation": abs(gen_val - oracle_val),
                "match": abs(gen_val - oracle_val) <= tolerance,
            }

        n_matched = sum(1 for d in deviations.values() if d["match"])
        return {
            "per_dimension": deviations,
            "n_matched": n_matched,
            "n_total": len(deviations),
            "all_matched": n_matched == len(deviations),
        }