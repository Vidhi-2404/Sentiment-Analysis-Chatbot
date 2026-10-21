"""
Sentiment Analysis Service using VADER (Valence Aware Dictionary and sEntiment Reasoner).
Provides sentiment classification (Positive, Neutral, Negative), confidence scoring,
ambiguity and mixed-sentiment detection, and tone adaptation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import (
    SENTIMENT_POSITIVE,
    SENTIMENT_NEUTRAL,
    SENTIMENT_NEGATIVE,
    COMPOUND_POS_THRESHOLD,
    COMPOUND_NEG_THRESHOLD,
    TONE_DIRECTIVES,
    CRISIS_KEYWORDS,
    CRISIS_RESPONSE,
)


@dataclass
class SentimentResult:
    """Structured result of sentiment classification."""
    label: str
    compound: float
    confidence: float
    detailed_scores: Dict[str, float]
    is_mixed: bool = False
    is_ambiguous: bool = False
    is_crisis: bool = False
    explanation: str = ""
    tone_directive: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "compound": round(self.compound, 4),
            "confidence": round(self.confidence, 4),
            "pos": round(self.detailed_scores.get("pos", 0.0), 4),
            "neu": round(self.detailed_scores.get("neu", 0.0), 4),
            "neg": round(self.detailed_scores.get("neg", 0.0), 4),
            "is_mixed": self.is_mixed,
            "is_ambiguous": self.is_ambiguous,
            "is_crisis": self.is_crisis,
            "explanation": self.explanation,
        }


class SentimentAnalyzer:
    """
    Lightweight, deterministic, and fast sentiment analyzer based on VADER.

    Model Strengths:
    - Optimized for conversational text, emoticons, emojis, punctuation emphasis, and capitalization.
    - Zero external network downloads or large checkpoints required; runs completely offline.
    - Instantaneous inference with deterministic scoring suitable for production Streamlit apps.

    Model Limitations:
    - Sarcasm and irony are difficult to recognize without contextual awareness.
    - Domain-specific idioms (e.g., medical test results: 'tumor is negative') can be inverted.
    - Complex syntactic inversion (e.g. double negatives) can occasionally misclassify.
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def check_crisis(self, text: str) -> bool:
        """Check if message contains distress or self-harm keywords requiring safety priority."""
        lower_text = text.lower()
        for kw in CRISIS_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", lower_text):
                return True
        return False

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze the sentiment of a given input text.
        Handles empty, ambiguous, and mixed sentiment cases gracefully.
        """
        if not text or not text.strip():
            return SentimentResult(
                label=SENTIMENT_NEUTRAL,
                compound=0.0,
                confidence=0.5,
                detailed_scores={"pos": 0.0, "neu": 1.0, "neg": 0.0, "compound": 0.0},
                is_ambiguous=True,
                explanation="Empty or whitespace-only input provided. Defaulted to Neutral.",
                tone_directive=TONE_DIRECTIVES[SENTIMENT_NEUTRAL],
            )

        cleaned_text = text.strip()
        is_crisis = self.check_crisis(cleaned_text)

        # Compute VADER polarity scores
        scores = self.analyzer.polarity_scores(cleaned_text)
        compound = float(scores["compound"])
        pos = float(scores["pos"])
        neu = float(scores["neu"])
        neg = float(scores["neg"])

        # Mixed sentiment detection
        # True if both positive and negative signals are meaningfully present,
        # or conjunctions like "but", "however", "although" show contrast.
        contrast_words = bool(re.search(r"\b(but|however|although|yet|though|despite)\b", cleaned_text, re.IGNORECASE))
        is_mixed = (pos >= 0.12 and neg >= 0.12) or (contrast_words and pos > 0.08 and neg > 0.08)

        # Ambiguity and hedging detection:
        # 1. Phrases with explicit hesitation or hedging markers
        hedging_pattern = r"\b(maybe|perhaps|i guess|suppose|hmm|ok|okay|fine|so-so|meh|whatever|it is what it is|not sure|hard to say|could be)\b"
        has_hedging = bool(re.search(hedging_pattern, cleaned_text, re.IGNORECASE))

        # 2. Borderline compound with high neutrality or hesitation
        is_ambiguous = (
            (abs(compound) < 0.15 and neu > 0.70)
            or (has_hedging and abs(compound) <= 0.35)
            or (len(cleaned_text.split()) <= 3 and abs(compound) <= 0.35)
        )

        # Classify label
        if is_ambiguous and abs(compound) < 0.30:
            # Borderline ambiguous responses are categorized as Neutral or flagged explicitly
            label = SENTIMENT_NEUTRAL
        elif compound >= COMPOUND_POS_THRESHOLD:
            label = SENTIMENT_POSITIVE
        elif compound <= COMPOUND_NEG_THRESHOLD:
            label = SENTIMENT_NEGATIVE
        else:
            label = SENTIMENT_NEUTRAL

        # Compute normalized confidence score [0.0 to 1.0]
        # For positive/negative, confidence increases with magnitude of compound and dominant polarity.
        # For neutral, confidence is highest when neutrality is undisputed.
        if label == SENTIMENT_POSITIVE:
            base_conf = 0.55 + 0.45 * compound
            if is_mixed:
                base_conf *= 0.82  # Diminish confidence for mixed feedback
            confidence = min(0.99, max(0.51, base_conf))
        elif label == SENTIMENT_NEGATIVE:
            base_conf = 0.55 + 0.45 * abs(compound)
            if is_mixed:
                base_conf *= 0.82
            confidence = min(0.99, max(0.51, base_conf))
        else:
            # Neutral confidence
            if is_ambiguous:
                confidence = 0.58
            else:
                confidence = min(0.95, max(0.50, 0.50 + 0.45 * neu))

        # Generate explanatory summary
        if is_crisis:
            explanation = "Safety protocol triggered: Distress or crisis keywords detected. Prioritizing support resources."
        elif is_mixed:
            explanation = f"Mixed sentiment detected: contains both positive ({pos:.0%}) and negative ({neg:.0%}) elements."
        elif is_ambiguous:
            explanation = "Subtle or ambiguous sentiment with minimal emotional valence detected."
        elif label == SENTIMENT_POSITIVE:
            explanation = f"Favorable sentiment identified (Compound: {compound:+.2f}, Positivity: {pos:.0%})."
        elif label == SENTIMENT_NEGATIVE:
            explanation = f"Critical or dissatisfied sentiment identified (Compound: {compound:+.2f}, Negativity: {neg:.0%})."
        else:
            explanation = f"Factual, neutral sentiment identified (Neutrality: {neu:.0%})."

        return SentimentResult(
            label=label,
            compound=compound,
            confidence=confidence,
            detailed_scores=scores,
            is_mixed=is_mixed,
            is_ambiguous=is_ambiguous,
            is_crisis=is_crisis,
            explanation=explanation,
            tone_directive=TONE_DIRECTIVES[label],
        )

    def adapt_response(self, base_response: str, sentiment: SentimentResult) -> str:
        """
        Adapt response style to match user emotional tone while strictly preserving
        factual correctness and safety precautions.
        """
        if sentiment.is_crisis:
            return f"{CRISIS_RESPONSE}\n\n---\n**Regarding your question:**\n{base_response}"

        tone = sentiment.tone_directive or TONE_DIRECTIVES[sentiment.label]
        prefix = tone.get("prefix", "")
        suffix = tone.get("suffix", "")

        # When the user's message is mixed, acknowledge nuance before answering
        if sentiment.is_mixed:
            prefix = "⚖️ I noticed you have mixed feelings here. Let's address this step-by-step: "

        # Compose adapted response without altering factual content
        return f"{prefix}{base_response.strip()}{suffix}"
