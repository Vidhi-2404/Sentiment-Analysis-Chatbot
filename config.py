"""
Configuration settings and constants for Sentiment Analysis Chatbot.
"""

import os
from pathlib import Path

# Application Metadata
APP_TITLE = "Sentiment Analysis Chatbot & Dashboard"
APP_SUBTITLE = "Real-time Sentiment Classification, Adaptive Responses & TF-IDF Knowledge Retrieval"
APP_VERSION = "1.0.0"

# Directories
BASE_DIR = Path(__file__).resolve().parent

# Sentiment Classification Constants
SENTIMENT_POSITIVE = "Positive"
SENTIMENT_NEUTRAL = "Neutral"
SENTIMENT_NEGATIVE = "Negative"

# VADER Compound Score Thresholds
# Standard VADER recommendations: >= 0.05 is positive, <= -0.05 is negative, otherwise neutral.
COMPOUND_POS_THRESHOLD = 0.05
COMPOUND_NEG_THRESHOLD = -0.05

# Color Palettes for UI and Plotly Charts
SENTIMENT_COLORS = {
    SENTIMENT_POSITIVE: "#10b981",  # Emerald Green
    SENTIMENT_NEUTRAL: "#6b7280",   # Cool Slate Gray
    SENTIMENT_NEGATIVE: "#ef4444",  # Crimson Red
}

SENTIMENT_EMOJIS = {
    SENTIMENT_POSITIVE: "😊",
    SENTIMENT_NEUTRAL: "😐",
    SENTIMENT_NEGATIVE: "😟",
}

# Adaptive Tone Directives
TONE_DIRECTIVES = {
    SENTIMENT_POSITIVE: {
        "badge": "Encouraging & Warm",
        "description": "Celebratory, appreciative, and enthusiastic.",
        "prefix": "🎉 Great to hear that! ",
        "suffix": " Let us know if there's anything else we can celebrate or explore!"
    },
    SENTIMENT_NEUTRAL: {
        "badge": "Objective & Informative",
        "description": "Clear, concise, balanced, and direct.",
        "prefix": "📌 Here is what you need to know: ",
        "suffix": " Feel free to ask any follow-up questions."
    },
    SENTIMENT_NEGATIVE: {
        "badge": "Empathetic & Solution-Oriented",
        "description": "Calm, validating, reassuring, and focused on resolution.",
        "prefix": "🤝 I understand this is frustrating, and I'm here to help resolve it. ",
        "suffix": " Please let me know if this solution works for you or if you'd like more options."
    }
}

# Safety & Medical Precaution Guardrails
SAFETY_DISCLAIMER = (
    "Disclaimer: Sentiment analysis is probabilistic and does not replace human judgment, "
    "legal counsel, or emergency medical guidance."
)

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self-harm", "hurt myself", "hopeless"
]

CRISIS_RESPONSE = (
    "⚠️ If you or someone you know is struggling or in crisis, help is available. "
    "You are not alone. Please reach out immediately:\n"
    "- In the US: Call or text 988, or chat at 988lifeline.org\n"
    "- In the UK: Call 111 or text SHOUT to 85258\n"
    "- In India: Call 9152987821 (Vandrevala Foundation) or 112\n"
    "- International: Visit https://findahelpline.com for local emergency resources."
)

# Retrieval Engine Defaults
DEFAULT_TOP_K = 2
MIN_RETRIEVAL_SCORE = 0.12
