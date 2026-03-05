"""
Emotion-GSRM: Evaluation Rubric
=================================
Seven sub-dimensions capturing emotional quality on a 1–5 Likert scale.

Each dimension has:
  - A structured definition with anchored scale descriptors
  - Acoustic feature associations (which features are most diagnostic)
  - Prompt templates for CoT synthesis stages
  - Scoring logic and validation

Dimensions:
  1. Emotional Intensity
  2. Emotional Appropriateness
  3. Emotional Consistency
  4. Valence Accuracy
  5. Arousal Accuracy
  6. Transition Smoothness
  7. Overall Emotional Quality
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class DimensionName(str, Enum):
    EMOTIONAL_INTENSITY = "emotional_intensity"
    EMOTIONAL_APPROPRIATENESS = "emotional_appropriateness"
    EMOTIONAL_CONSISTENCY = "emotional_consistency"
    VALENCE_ACCURACY = "valence_accuracy"
    AROUSAL_ACCURACY = "arousal_accuracy"
    TRANSITION_SMOOTHNESS = "transition_smoothness"
    OVERALL_EMOTIONAL_QUALITY = "overall_emotional_quality"


SCORE_RANGE = (1, 5)  # Likert scale bounds


# ---------------------------------------------------------------------------
# Scale Anchor Descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScaleAnchor:
    """Descriptor for a single point on the Likert scale."""
    score: int
    label: str
    description: str


@dataclass(frozen=True)
class RubricDimension:
    """
    Full definition of one evaluation sub-dimension.

    Attributes
    ----------
    name : DimensionName
        Unique identifier for this dimension.
    display_name : str
        Human-readable name.
    definition : str
        What this dimension measures.
    anchors : tuple of ScaleAnchor
        Scale point descriptors (1 through 5).
    associated_features : tuple of str
        Acoustic features most diagnostic for this dimension.
    evidence_prompt : str
        Prompt template for Stage 1 evidence generation.
    judgment_prompt : str
        Prompt template for Stage 2 CoT judgment.
    """
    name: DimensionName
    display_name: str
    definition: str
    anchors: tuple[ScaleAnchor, ...]
    associated_features: tuple[str, ...]
    evidence_prompt: str
    judgment_prompt: str


# ---------------------------------------------------------------------------
# Rubric Definitions
# ---------------------------------------------------------------------------

EMOTIONAL_INTENSITY = RubricDimension(
    name=DimensionName.EMOTIONAL_INTENSITY,
    display_name="Emotional Intensity",
    definition="Strength and vividness of emotional expression, from flat/monotone to vivid/expressive.",
    anchors=(
        ScaleAnchor(1, "Flat", "No discernible emotional expression; completely monotone delivery with minimal pitch variation, narrow intensity range, and no voice quality modulation."),
        ScaleAnchor(2, "Subdued", "Slight emotional coloring but largely restrained; minor pitch movement and limited dynamic range."),
        ScaleAnchor(3, "Moderate", "Clearly present emotional expression with noticeable prosodic variation; appropriate for neutral conversational contexts."),
        ScaleAnchor(4, "Expressive", "Strong emotional delivery with wide pitch range, dynamic intensity, and clear voice quality changes that convey the intended emotion."),
        ScaleAnchor(5, "Vivid", "Highly expressive with maximal appropriate use of prosodic, voice quality, and temporal cues; emotionally compelling and engaging."),
    ),
    associated_features=(
        "pitch_variation_mean", "intensity_variation_mean",
        "jitter", "shimmer", "speech_rate_variation",
        "vowel_space_area",
    ),
    evidence_prompt=(
        "Assess the emotional intensity of this utterance. Examine pitch variation "
        "(wide range suggests high intensity), intensity dynamics, voice quality "
        "perturbations (jitter/shimmer), and vowel space expansion. A flat delivery "
        "will show narrow pitch range and low variation across features."
    ),
    judgment_prompt=(
        "Rate emotional intensity on 1-5. Connect specific acoustic evidence to your "
        "rating: what pitch, intensity, and voice quality patterns indicate the level "
        "of emotional vividness?"
    ),
)

EMOTIONAL_APPROPRIATENESS = RubricDimension(
    name=DimensionName.EMOTIONAL_APPROPRIATENESS,
    display_name="Emotional Appropriateness",
    definition="Whether the expressed emotion fits the conversational context and the user's emotional state.",
    anchors=(
        ScaleAnchor(1, "Inappropriate", "Emotion is completely mismatched to context (e.g., cheerful delivery for condolences, aggressive tone for reassurance)."),
        ScaleAnchor(2, "Poorly matched", "Emotion partially conflicts with context; may get valence right but with wrong intensity or style."),
        ScaleAnchor(3, "Acceptable", "Emotion is generally appropriate but generic; no major mismatches but lacks nuanced adaptation to context."),
        ScaleAnchor(4, "Well-matched", "Emotion clearly fits the conversational context with appropriate intensity and style adjustments."),
        ScaleAnchor(5, "Perfectly attuned", "Emotion is precisely calibrated to the context, user state, and conversational dynamics; demonstrates sophisticated emotional intelligence."),
    ),
    associated_features=(
        "pitch_level_mean", "pitch_slope_mean",
        "speech_rate", "spectral_tilt",
        "pause_duration_mean", "hnr",
    ),
    evidence_prompt=(
        "Evaluate whether the emotional delivery matches the conversational context. "
        "Consider: Does the pitch contour (rising for questions, falling for statements "
        "of sympathy) match what the context requires? Does the speech rate align with "
        "the expected emotional register? Examine the transcript and preceding turns."
    ),
    judgment_prompt=(
        "Rate emotional appropriateness on 1-5. Ground your judgment in specific "
        "mismatches or alignments between the acoustic delivery and what the "
        "conversational context demands."
    ),
)

EMOTIONAL_CONSISTENCY = RubricDimension(
    name=DimensionName.EMOTIONAL_CONSISTENCY,
    display_name="Emotional Consistency",
    definition="Coherence of emotional tone within a single response; absence of jarring emotional shifts.",
    anchors=(
        ScaleAnchor(1, "Incoherent", "Emotional tone changes erratically with no discernible pattern; feels like multiple speakers spliced together."),
        ScaleAnchor(2, "Inconsistent", "Noticeable emotional shifts that disrupt coherence; tone drifts without clear motivation."),
        ScaleAnchor(3, "Mostly consistent", "Generally stable emotional tone with minor fluctuations that don't significantly disrupt the message."),
        ScaleAnchor(4, "Consistent", "Stable emotional tone throughout with only motivated variations that serve the message."),
        ScaleAnchor(5, "Perfectly coherent", "Seamless emotional arc that maintains or evolves the emotional tone in a completely natural and motivated way."),
    ),
    associated_features=(
        "pitch_variation_mean", "intensity_variation_mean",
        "speech_rate_variation", "spectral_tilt",
    ),
    evidence_prompt=(
        "Analyze the consistency of emotional expression across the utterance. "
        "Look for abrupt changes in pitch level, intensity, or speech rate that "
        "aren't motivated by the content. Compare prosodic features across the "
        "first and second halves of the utterance."
    ),
    judgment_prompt=(
        "Rate emotional consistency on 1-5. Identify any unmotivated shifts in "
        "vocal parameters and explain whether variations serve the message or "
        "disrupt emotional coherence."
    ),
)

VALENCE_ACCURACY = RubricDimension(
    name=DimensionName.VALENCE_ACCURACY,
    display_name="Valence Accuracy",
    definition="Alignment of positive/negative emotional coloring with the semantic content.",
    anchors=(
        ScaleAnchor(1, "Inverted", "Emotional valence is opposite to content (e.g., happy-sounding delivery of sad content)."),
        ScaleAnchor(2, "Misaligned", "Valence direction has noticeable errors; tone sometimes contradicts the message."),
        ScaleAnchor(3, "Neutral/ambiguous", "Valence is not clearly positive or negative; acceptable for neutral content but lacks specificity."),
        ScaleAnchor(4, "Aligned", "Positive/negative coloring clearly matches the semantic content with appropriate acoustic cues."),
        ScaleAnchor(5, "Precisely matched", "Valence is perfectly calibrated with nuanced acoustic markers (e.g., warmth for positive, compressed spectral tilt for negative)."),
    ),
    associated_features=(
        "pitch_level_mean", "spectral_tilt", "hnr",
        "f1_mean", "f2_mean", "speech_rate",
    ),
    evidence_prompt=(
        "Assess the valence alignment between vocal expression and semantic content. "
        "Positive emotions tend to correlate with higher pitch, brighter spectral tilt, "
        "higher HNR, and expanded vowel space. Negative emotions often show lower pitch, "
        "compressed spectral tilt, and reduced vowel space. Compare these patterns to "
        "the content's emotional valence."
    ),
    judgment_prompt=(
        "Rate valence accuracy on 1-5. Explain which acoustic features indicate "
        "positive or negative coloring and whether this matches the semantic content."
    ),
)

AROUSAL_ACCURACY = RubricDimension(
    name=DimensionName.AROUSAL_ACCURACY,
    display_name="Arousal Accuracy",
    definition="Whether the energy level matches the intended activation level of the content.",
    anchors=(
        ScaleAnchor(1, "Completely wrong", "Energy level is opposite to what content requires (e.g., lethargic delivery of urgent news)."),
        ScaleAnchor(2, "Mismatched", "Energy level is noticeably off; too high or too low for the content's activation demands."),
        ScaleAnchor(3, "Approximate", "Energy level is in the right general range but not precisely calibrated."),
        ScaleAnchor(4, "Well-calibrated", "Energy level clearly matches the activation requirements of the content."),
        ScaleAnchor(5, "Precisely calibrated", "Energy level is perfectly tuned with appropriate speech rate, intensity, and temporal dynamics for the content."),
    ),
    associated_features=(
        "speech_rate", "intensity_level_mean",
        "pitch_variation_mean", "jitter", "shimmer",
        "pause_frequency",
    ),
    evidence_prompt=(
        "Evaluate whether the vocal energy matches the content's activation level. "
        "High arousal correlates with faster speech rate, higher intensity, wider "
        "pitch variation, and increased jitter/shimmer. Low arousal shows slower "
        "rate, lower intensity, and longer pauses. Compare the acoustic energy "
        "profile to what the content demands."
    ),
    judgment_prompt=(
        "Rate arousal accuracy on 1-5. Map specific acoustic energy indicators "
        "to the content's activation requirements and explain any mismatches."
    ),
)

TRANSITION_SMOOTHNESS = RubricDimension(
    name=DimensionName.TRANSITION_SMOOTHNESS,
    display_name="Transition Smoothness",
    definition="Naturalness of emotional shifts across conversational turns.",
    anchors=(
        ScaleAnchor(1, "Jarring", "Emotional transitions are abrupt and unnatural; feels like emotional register resets between turns."),
        ScaleAnchor(2, "Awkward", "Transitions are noticeable and somewhat disruptive; emotional shifts happen too fast or too slow."),
        ScaleAnchor(3, "Acceptable", "Transitions are functional but lack subtlety; emotional shifts are present but mechanical."),
        ScaleAnchor(4, "Smooth", "Emotional shifts between turns are natural with appropriate bridging cues (pause, pitch reset)."),
        ScaleAnchor(5, "Seamless", "Emotional transitions are imperceptibly smooth; emotional arc across turns feels completely natural and human-like."),
    ),
    associated_features=(
        "pitch_slope_mean", "pause_duration_mean",
        "speech_rate", "speech_rate_variation",
        "intensity_level_mean",
    ),
    evidence_prompt=(
        "Evaluate the naturalness of emotional transitions relative to the preceding "
        "conversational turns. Look for bridging cues: appropriate pauses before "
        "emotional shifts, gradual pitch adjustments, and smooth intensity changes. "
        "Abrupt resets in any parameter suggest poor transitions."
    ),
    judgment_prompt=(
        "Rate transition smoothness on 1-5. Focus on the emotional continuity "
        "between this utterance and the preceding context. Cite specific acoustic "
        "bridging cues or their absence."
    ),
)

OVERALL_EMOTIONAL_QUALITY = RubricDimension(
    name=DimensionName.OVERALL_EMOTIONAL_QUALITY,
    display_name="Overall Emotional Quality",
    definition="Holistic judgment of emotional expressivity across all dimensions.",
    anchors=(
        ScaleAnchor(1, "Poor", "Emotionally unacceptable; multiple dimensions fail simultaneously, creating a fundamentally broken emotional delivery."),
        ScaleAnchor(2, "Below average", "Significant emotional deficiencies across multiple dimensions; delivery is tolerable but unsatisfying."),
        ScaleAnchor(3, "Average", "Emotionally functional; no major failures but lacks distinction or emotional engagement."),
        ScaleAnchor(4, "Good", "Emotionally effective; strong performance across most dimensions with minor areas for improvement."),
        ScaleAnchor(5, "Excellent", "Emotionally outstanding; human-level or near-human emotional expressivity across all dimensions."),
    ),
    associated_features=(
        "pitch_level_mean", "pitch_variation_mean", "intensity_level_mean",
        "jitter", "shimmer", "hnr", "spectral_tilt",
        "speech_rate", "speech_rate_variation",
        "pause_duration_mean", "f1_mean", "f2_mean", "vowel_space_area",
    ),
    evidence_prompt=(
        "Provide a holistic assessment of emotional quality considering all acoustic "
        "features and the previous sub-dimension assessments. Weigh the relative "
        "importance of each dimension for this particular conversational context."
    ),
    judgment_prompt=(
        "Rate overall emotional quality on 1-5. Synthesize evidence from all "
        "sub-dimensions into a coherent holistic judgment. Explain which dimensions "
        "contributed most to the final rating and why."
    ),
)


# ---------------------------------------------------------------------------
# Rubric Registry
# ---------------------------------------------------------------------------

RUBRIC_DIMENSIONS: dict[DimensionName, RubricDimension] = {
    DimensionName.EMOTIONAL_INTENSITY: EMOTIONAL_INTENSITY,
    DimensionName.EMOTIONAL_APPROPRIATENESS: EMOTIONAL_APPROPRIATENESS,
    DimensionName.EMOTIONAL_CONSISTENCY: EMOTIONAL_CONSISTENCY,
    DimensionName.VALENCE_ACCURACY: VALENCE_ACCURACY,
    DimensionName.AROUSAL_ACCURACY: AROUSAL_ACCURACY,
    DimensionName.TRANSITION_SMOOTHNESS: TRANSITION_SMOOTHNESS,
    DimensionName.OVERALL_EMOTIONAL_QUALITY: OVERALL_EMOTIONAL_QUALITY,
}


# ---------------------------------------------------------------------------
# Score Container
# ---------------------------------------------------------------------------

@dataclass
class EmotionScores:
    """
    Container for scores across all 7 sub-dimensions.

    Validates that all scores are within the 1-5 Likert range.
    """
    emotional_intensity: float
    emotional_appropriateness: float
    emotional_consistency: float
    valence_accuracy: float
    arousal_accuracy: float
    transition_smoothness: float
    overall_emotional_quality: float
    utterance_id: str = ""
    annotator_id: str = ""

    def __post_init__(self):
        for dim_name in DimensionName:
            attr = dim_name.value
            val = getattr(self, attr)
            if not (SCORE_RANGE[0] <= val <= SCORE_RANGE[1]):
                raise ValueError(
                    f"Score for {attr} must be between {SCORE_RANGE[0]} and "
                    f"{SCORE_RANGE[1]}, got {val}"
                )

    def to_dict(self) -> dict[str, float]:
        """Return scores as a dimension_name -> score mapping."""
        return {
            dim_name.value: getattr(self, dim_name.value)
            for dim_name in DimensionName
        }

    def mean_score(self) -> float:
        """Average across all 7 dimensions."""
        scores = self.to_dict()
        return sum(scores.values()) / len(scores)

    @classmethod
    def from_dict(
        cls, scores: dict[str, float],
        utterance_id: str = "",
        annotator_id: str = "",
    ) -> "EmotionScores":
        """Construct from a dictionary of dimension_name -> score."""
        return cls(
            utterance_id=utterance_id,
            annotator_id=annotator_id,
            **{dim.value: scores[dim.value] for dim in DimensionName},
        )


# ---------------------------------------------------------------------------
# Score Aggregation (multi-annotator)
# ---------------------------------------------------------------------------

def aggregate_scores(
    score_lists: list[EmotionScores],
    method: str = "mean",
) -> EmotionScores:
    """
    Aggregate multiple annotator scores into a single consensus score.

    Parameters
    ----------
    score_lists : list of EmotionScores
        Scores from multiple annotators for the same utterance.
    method : str
        Aggregation method: 'mean' or 'median'.

    Returns
    -------
    EmotionScores
        Aggregated consensus scores.
    """
    import numpy as np

    if not score_lists:
        raise ValueError("Cannot aggregate empty score list")

    agg_fn = np.mean if method == "mean" else np.median
    aggregated = {}

    for dim_name in DimensionName:
        values = [getattr(s, dim_name.value) for s in score_lists]
        raw = float(agg_fn(values))
        aggregated[dim_name.value] = round(max(1.0, min(5.0, raw)), 2)

    return EmotionScores.from_dict(
        aggregated,
        utterance_id=score_lists[0].utterance_id,
        annotator_id=f"aggregated_{method}_{len(score_lists)}",
    )


def inter_annotator_agreement(
    score_lists: list[EmotionScores],
) -> dict[str, float]:
    """
    Compute inter-annotator agreement per dimension using ICC(2,1).

    Falls back to Pearson correlation for 2 annotators.

    Parameters
    ----------
    score_lists : list of EmotionScores
        Scores from multiple annotators for multiple utterances,
        grouped by utterance (outer list) containing annotator scores
        (inner items).

    Returns
    -------
    dict
        Dimension name -> agreement score.
    """
    import numpy as np

    agreement = {}
    n_annotators = len(score_lists[0].to_dict()) if score_lists else 0

    for dim_name in DimensionName:
        values = [getattr(s, dim_name.value) for s in score_lists]
        if len(values) < 2:
            agreement[dim_name.value] = 1.0
            continue

        # Simple pairwise correlation as proxy
        arr = np.array(values)
        overall_mean = np.mean(arr)
        variance = np.var(arr)
        agreement[dim_name.value] = 1.0 - (variance / max(variance + 1e-8, 1.0))

    return agreement


# ---------------------------------------------------------------------------
# Rubric Prompt Builder
# ---------------------------------------------------------------------------

class RubricPromptBuilder:
    """
    Constructs evaluation prompts incorporating rubric definitions.

    Used during CoT synthesis to guide GPT-4o (teacher model) and
    during inference to structure Qwen2.5-Omni-7B's reasoning.
    """

    @staticmethod
    def build_rubric_description(
        dimensions: Optional[list[DimensionName]] = None,
    ) -> str:
        """
        Generate a formatted rubric description for prompt injection.

        Parameters
        ----------
        dimensions : list of DimensionName, optional
            Subset of dimensions to include. Defaults to all 7.
        """
        if dimensions is None:
            dimensions = list(DimensionName)

        lines = [
            "=== Emotion Evaluation Rubric ===",
            "Rate each dimension on a 1-5 Likert scale.",
            "",
        ]

        for dim_name in dimensions:
            dim = RUBRIC_DIMENSIONS[dim_name]
            lines.append(f"### {dim.display_name}")
            lines.append(f"Definition: {dim.definition}")
            lines.append("Scale:")
            for anchor in dim.anchors:
                lines.append(f"  {anchor.score} ({anchor.label}): {anchor.description}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def build_evidence_prompt(
        dimension: DimensionName,
        acoustic_features_text: str,
        transcript: str,
        context_turns: Optional[list[str]] = None,
    ) -> str:
        """
        Build the Stage 1 evidence generation prompt for a single dimension.

        Parameters
        ----------
        dimension : DimensionName
            Which dimension to generate evidence for.
        acoustic_features_text : str
            Formatted acoustic features (from format_features_for_prompt).
        transcript : str
            The utterance transcript.
        context_turns : list of str, optional
            Preceding 2-3 conversational turns for context.
        """
        dim = RUBRIC_DIMENSIONS[dimension]
        context_block = ""
        if context_turns:
            formatted_turns = "\n".join(
                f"  Turn {i+1}: {turn}" for i, turn in enumerate(context_turns)
            )
            context_block = f"\nConversational Context (preceding turns):\n{formatted_turns}\n"

        return (
            f"You are evaluating the {dim.display_name} of a speech utterance.\n"
            f"\n"
            f"Definition: {dim.definition}\n"
            f"{context_block}\n"
            f"Transcript: \"{transcript}\"\n"
            f"\n"
            f"{acoustic_features_text}\n"
            f"\n"
            f"Task: {dim.evidence_prompt}\n"
            f"\n"
            f"Provide your analysis in the following structure:\n"
            f"1. Inferred emotional context from transcript and preceding turns\n"
            f"2. Detected emotional cues from acoustic features (cite specific values)\n"
            f"3. {dim.display_name} assessment with supporting evidence\n"
            f"4. Strengths of the emotional delivery for this dimension\n"
            f"5. Issues or areas of concern\n"
        )

    @staticmethod
    def build_judgment_prompt(
        evidence_log: str,
        oracle_scores: Optional[dict[str, float]] = None,
    ) -> str:
        """
        Build the Stage 2 global judgment CoT prompt.

        Parameters
        ----------
        evidence_log : str
            Combined Stage 1 evidence across all dimensions.
        oracle_scores : dict, optional
            Target scores to match during CoT synthesis (training only).
            At inference time, this should be None.
        """
        rubric_text = RubricPromptBuilder.build_rubric_description()

        oracle_block = ""
        if oracle_scores:
            scores_text = "\n".join(
                f"  {name}: {score}" for name, score in oracle_scores.items()
            )
            oracle_block = (
                f"\nTarget scores (your reasoning must arrive at these naturally):\n"
                f"{scores_text}\n"
            )

        return (
            f"You are synthesizing a global emotional quality judgment from "
            f"detailed evidence.\n"
            f"\n"
            f"{rubric_text}\n"
            f"\n"
            f"=== Evidence Log ===\n"
            f"{evidence_log}\n"
            f"{oracle_block}\n"
            f"Task: Synthesize a coherent chain-of-thought assessment that:\n"
            f"1. Connects acoustic evidence to emotion scores across all 7 dimensions\n"
            f"2. Identifies cross-dimensional patterns (e.g., high intensity but poor appropriateness)\n"
            f"3. Weighs dimension importance for this specific context\n"
            f"4. Produces final scores for each dimension with clear justification\n"
            f"\n"
            f"Format your response as:\n"
            f"[REASONING]\n"
            f"<your chain-of-thought analysis>\n"
            f"[SCORES]\n"
            f"emotional_intensity: <score>\n"
            f"emotional_appropriateness: <score>\n"
            f"emotional_consistency: <score>\n"
            f"valence_accuracy: <score>\n"
            f"arousal_accuracy: <score>\n"
            f"transition_smoothness: <score>\n"
            f"overall_emotional_quality: <score>\n"
        )

    @staticmethod
    def parse_scores_from_response(response_text: str) -> dict[str, float]:
        """
        Parse dimension scores from a model's CoT response.

        Expects the [SCORES] section format produced by build_judgment_prompt.
        """
        scores = {}
        in_scores = False

        for line in response_text.split("\n"):
            line = line.strip()
            if line == "[SCORES]":
                in_scores = True
                continue
            if in_scores and ":" in line:
                parts = line.split(":", 1)
                dim_name = parts[0].strip()
                try:
                    score = float(parts[1].strip())
                    score = max(1.0, min(5.0, score))
                    scores[dim_name] = score
                except ValueError:
                    continue

        # Validate all dimensions present
        expected = {d.value for d in DimensionName}
        missing = expected - set(scores.keys())
        if missing:
            for m in missing:
                scores[m] = 3.0  # Default to neutral

        return scores