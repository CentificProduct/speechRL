"""
Emotion-GSRM: Stage 1 — Utterance-Level Emotion Evidence Log
==============================================================
For each utterance, GPT-4o receives the transcript, acoustic features,
and conversational context (preceding 2-3 turns). It generates:

  1. Inferred emotional context
  2. Detected emotional cues (grounded in specific acoustic features)
  3. Emotional alignment assessment
  4. Strengths of the delivery
  5. Issues or concerns

This follows the GSRM two-stage synthesis architecture, adapted for
emotion-specific evaluation dimensions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from dataloader.base import EmotionSample, CategoricalEmotion
from features.acoustic import AcousticFeatureSet
from rubric.init import (
    DimensionName,
    RUBRIC_DIMENSIONS,
    RubricPromptBuilder,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evidence Log Data Structure
# ---------------------------------------------------------------------------

@dataclass
class DimensionEvidence:
    """Evidence for a single evaluation dimension."""
    dimension: str
    emotional_context: str
    detected_cues: str
    alignment_assessment: str
    strengths: str
    issues: str
    raw_response: str = ""


@dataclass
class EvidenceLog:
    """
    Complete utterance-level evidence log across all dimensions.

    This is the output of Stage 1 and the input to Stage 2.
    """
    utterance_id: str
    transcript: str
    context_turns: list[str]
    acoustic_features_text: str
    dimension_evidence: dict[str, DimensionEvidence] = field(default_factory=dict)
    generation_metadata: dict = field(default_factory=dict)

    def to_prompt_text(self) -> str:
        """Format the full evidence log for Stage 2 input."""
        lines = [
            f"=== Evidence Log for Utterance: {self.utterance_id} ===",
            f"Transcript: \"{self.transcript}\"",
        ]

        if self.context_turns:
            lines.append("\nConversational Context:")
            for turn in self.context_turns:
                lines.append(f"  {turn}")

        lines.append(f"\n{self.acoustic_features_text}")
        lines.append("\n--- Dimension-Level Evidence ---")

        for dim_name, evidence in self.dimension_evidence.items():
            display_name = RUBRIC_DIMENSIONS.get(
                DimensionName(dim_name),
                type("", (), {"display_name": dim_name})(),
            ).display_name
            lines.extend([
                f"\n## {display_name}",
                f"Emotional Context: {evidence.emotional_context}",
                f"Detected Cues: {evidence.detected_cues}",
                f"Alignment Assessment: {evidence.alignment_assessment}",
                f"Strengths: {evidence.strengths}",
                f"Issues: {evidence.issues}",
            ])

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "utterance_id": self.utterance_id,
            "transcript": self.transcript,
            "context_turns": self.context_turns,
            "acoustic_features_text": self.acoustic_features_text,
            "dimension_evidence": {
                k: {
                    "dimension": v.dimension,
                    "emotional_context": v.emotional_context,
                    "detected_cues": v.detected_cues,
                    "alignment_assessment": v.alignment_assessment,
                    "strengths": v.strengths,
                    "issues": v.issues,
                }
                for k, v in self.dimension_evidence.items()
            },
            "generation_metadata": self.generation_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvidenceLog":
        """Deserialize from storage."""
        log = cls(
            utterance_id=data["utterance_id"],
            transcript=data["transcript"],
            context_turns=data.get("context_turns", []),
            acoustic_features_text=data.get("acoustic_features_text", ""),
            generation_metadata=data.get("generation_metadata", {}),
        )
        for dim_name, ev_data in data.get("dimension_evidence", {}).items():
            log.dimension_evidence[dim_name] = DimensionEvidence(**ev_data)
        return log


# ---------------------------------------------------------------------------
# Evidence Generator (Stage 1)
# ---------------------------------------------------------------------------

# System prompt for GPT-4o teacher model
STAGE1_SYSTEM_PROMPT = """\
You are an expert speech emotion analyst evaluating the emotional \
expressivity of a speech utterance. You will receive:
1. The utterance transcript
2. Conversational context (preceding turns)
3. Quantified acoustic features with ordinal categories

Your task is to generate a detailed evidence log analyzing the emotional \
quality of the speech delivery for a specific evaluation dimension. \
Ground ALL observations in specific acoustic feature values provided.

Respond in EXACTLY this JSON format:
{
    "emotional_context": "...",
    "detected_cues": "...",
    "alignment_assessment": "...",
    "strengths": "...",
    "issues": "..."
}

Rules:
- ALWAYS cite specific acoustic feature values (e.g., "pitch variation is \
high at 45.2 Hz std")
- Connect acoustic evidence to emotional interpretation
- Be specific about what emotion the delivery conveys
- Compare delivery to what the context demands
- Do NOT hallucinate features not provided in the input
"""


class EvidenceGenerator:
    """
    Stage 1: Generate utterance-level emotion evidence logs.

    Uses GPT-4o as the teacher model to analyze acoustic features
    and produce structured evidence for each evaluation dimension.

    Parameters
    ----------
    api_key : str
        OpenAI API key.
    model : str
        Teacher model name (default: "gpt-4o").
    temperature : float
        Sampling temperature (default: 0.3 for consistency).
    max_retries : int
        Max API call retries on failure (default: 3).
    retry_delay : float
        Seconds between retries (default: 1.0).
    dimensions : list of DimensionName, optional
        Which dimensions to generate evidence for. Defaults to all 7.
    rate_limit_delay : float
        Minimum seconds between API calls (default: 0.1).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        dimensions: Optional[list[DimensionName]] = None,
        rate_limit_delay: float = 0.1,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.dimensions = dimensions or list(DimensionName)
        self.rate_limit_delay = rate_limit_delay

        self._client = None
        self._call_count = 0
        self._last_call_time = 0.0

    @property
    def client(self):
        """Lazy-initialize OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: "
                    "pip install openai"
                )
        return self._client

    def generate_evidence(
        self,
        sample: EmotionSample,
        acoustic_features_text: str,
    ) -> EvidenceLog:
        """
        Generate a complete evidence log for one utterance.

        Calls GPT-4o once per dimension, collecting structured evidence.

        Parameters
        ----------
        sample : EmotionSample
            The utterance to analyze.
        acoustic_features_text : str
            Formatted acoustic features from format_features_for_prompt().

        Returns
        -------
        EvidenceLog
            Complete evidence across all configured dimensions.
        """
        evidence_log = EvidenceLog(
            utterance_id=sample.utterance_id,
            transcript=sample.transcript,
            context_turns=sample.context_turns,
            acoustic_features_text=acoustic_features_text,
        )

        start_time = time.time()

        for dim in self.dimensions:
            logger.debug(
                f"Generating evidence for {sample.utterance_id} / {dim.value}"
            )

            prompt = RubricPromptBuilder.build_evidence_prompt(
                dimension=dim,
                acoustic_features_text=acoustic_features_text,
                transcript=sample.transcript,
                context_turns=sample.context_turns,
            )

            response = self._call_teacher(prompt)
            evidence = self._parse_evidence_response(response, dim.value)
            evidence_log.dimension_evidence[dim.value] = evidence

        elapsed = time.time() - start_time
        evidence_log.generation_metadata = {
            "model": self.model,
            "temperature": self.temperature,
            "n_dimensions": len(self.dimensions),
            "elapsed_seconds": round(elapsed, 2),
            "api_calls": len(self.dimensions),
        }

        logger.info(
            f"Evidence log generated for {sample.utterance_id} "
            f"({len(self.dimensions)} dimensions, {elapsed:.1f}s)"
        )

        return evidence_log

    def generate_evidence_batch(
        self,
        sample: EmotionSample,
        acoustic_features_text: str,
    ) -> EvidenceLog:
        """
        Generate evidence for ALL dimensions in a single API call.

        More token-efficient than per-dimension calls, but may produce
        less detailed evidence. Use for large-scale synthesis.
        """
        evidence_log = EvidenceLog(
            utterance_id=sample.utterance_id,
            transcript=sample.transcript,
            context_turns=sample.context_turns,
            acoustic_features_text=acoustic_features_text,
        )

        # Build a combined prompt for all dimensions
        rubric_text = RubricPromptBuilder.build_rubric_description(self.dimensions)
        context_block = ""
        if sample.context_turns:
            turns = "\n".join(f"  {t}" for t in sample.context_turns)
            context_block = f"\nConversational Context:\n{turns}\n"

        prompt = (
            f"Analyze the emotional expressivity of this utterance across "
            f"all evaluation dimensions.\n\n"
            f"Transcript: \"{sample.transcript}\"\n"
            f"{context_block}\n"
            f"{acoustic_features_text}\n\n"
            f"{rubric_text}\n\n"
            f"For EACH dimension, provide analysis in this JSON format:\n"
            f"{{\n"
            f"  \"dimension_name\": {{\n"
            f"    \"emotional_context\": \"...\",\n"
            f"    \"detected_cues\": \"...\",\n"
            f"    \"alignment_assessment\": \"...\",\n"
            f"    \"strengths\": \"...\",\n"
            f"    \"issues\": \"...\"\n"
            f"  }},\n"
            f"  ...\n"
            f"}}\n\n"
            f"Cover all 7 dimensions. Ground ALL observations in specific "
            f"acoustic feature values."
        )

        start_time = time.time()
        response = self._call_teacher(prompt)
        elapsed = time.time() - start_time

        # Parse batch response
        self._parse_batch_response(response, evidence_log)

        evidence_log.generation_metadata = {
            "model": self.model,
            "temperature": self.temperature,
            "n_dimensions": len(self.dimensions),
            "elapsed_seconds": round(elapsed, 2),
            "api_calls": 1,
            "batch_mode": True,
        }

        return evidence_log

    def _call_teacher(self, user_prompt: str) -> str:
        """
        Call GPT-4o teacher model with retry logic.

        Returns the raw response text.
        """
        # Rate limiting
        now = time.time()
        elapsed_since_last = now - self._last_call_time
        if elapsed_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed_since_last)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=1500,
                )
                self._call_count += 1
                self._last_call_time = time.time()

                return response.choices[0].message.content or ""

            except Exception as e:
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    logger.error(f"All retries exhausted for teacher call")
                    return ""

        return ""

    def _parse_evidence_response(
        self, response: str, dimension: str
    ) -> DimensionEvidence:
        """Parse a single-dimension evidence response from GPT-4o."""
        try:
            # Strip markdown code fences if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)

            return DimensionEvidence(
                dimension=dimension,
                emotional_context=data.get("emotional_context", ""),
                detected_cues=data.get("detected_cues", ""),
                alignment_assessment=data.get("alignment_assessment", ""),
                strengths=data.get("strengths", ""),
                issues=data.get("issues", ""),
                raw_response=response,
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(
                f"Failed to parse evidence for {dimension}: {e}. "
                f"Using raw response as fallback."
            )
            return DimensionEvidence(
                dimension=dimension,
                emotional_context=response[:500] if response else "",
                detected_cues="",
                alignment_assessment="",
                strengths="",
                issues="",
                raw_response=response,
            )

    def _parse_batch_response(
        self, response: str, evidence_log: EvidenceLog
    ) -> None:
        """Parse a batch response covering all dimensions."""
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)

            for dim_name in self.dimensions:
                key = dim_name.value
                # Try both the enum value and display name as keys
                dim_data = data.get(key, data.get(
                    RUBRIC_DIMENSIONS[dim_name].display_name, {}
                ))

                if isinstance(dim_data, dict):
                    evidence_log.dimension_evidence[key] = DimensionEvidence(
                        dimension=key,
                        emotional_context=dim_data.get("emotional_context", ""),
                        detected_cues=dim_data.get("detected_cues", ""),
                        alignment_assessment=dim_data.get("alignment_assessment", ""),
                        strengths=dim_data.get("strengths", ""),
                        issues=dim_data.get("issues", ""),
                        raw_response="",
                    )
                else:
                    evidence_log.dimension_evidence[key] = DimensionEvidence(
                        dimension=key,
                        emotional_context="",
                        detected_cues="",
                        alignment_assessment="",
                        strengths="",
                        issues="",
                    )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse batch evidence: {e}")
            for dim_name in self.dimensions:
                if dim_name.value not in evidence_log.dimension_evidence:
                    evidence_log.dimension_evidence[dim_name.value] = DimensionEvidence(
                        dimension=dim_name.value,
                        emotional_context="Parse error - see raw response",
                        detected_cues="",
                        alignment_assessment="",
                        strengths="",
                        issues="",
                        raw_response=response[:500],
                    )