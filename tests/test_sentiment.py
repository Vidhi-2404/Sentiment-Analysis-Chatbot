"""
Unit and integration tests for Sentiment Analysis, Retrieval, and Tone Adaptation.
Run with: pytest
"""

import pytest
from unittest.mock import MagicMock, patch

from config import (
    SENTIMENT_POSITIVE,
    SENTIMENT_NEUTRAL,
    SENTIMENT_NEGATIVE,
    CRISIS_KEYWORDS,
)
from services.sentiment_service import SentimentAnalyzer, SentimentResult
from services.retrieval_service import RetrievalEngine
from services.gemini_service import GeminiService


@pytest.fixture
def analyzer():
    return SentimentAnalyzer()


@pytest.fixture
def retrieval_engine():
    return RetrievalEngine()


# -----------------------------------------------------------------------------
# 1. Classification & Evaluation on Labeled Test Set
# -----------------------------------------------------------------------------

def test_positive_classification(analyzer):
    positive_samples = [
        "I absolutely love this new update! Everything works smoothly and looks wonderful.",
        "Antigravity and Streamlit are fantastic tools for rapid AI development! 🚀",
        "The customer service representative was super kind, quick, and helpful.",
    ]
    for text in positive_samples:
        result = analyzer.analyze(text)
        assert result.label == SENTIMENT_POSITIVE, f"Expected Positive for: {text}"
        assert result.compound > 0.05
        assert 0.0 <= result.confidence <= 1.0


def test_negative_classification(analyzer):
    negative_samples = [
        "This is completely unacceptable, utterly broken, and a huge waste of my time.",
        "Terrible experience. The system crashed twice and corrupted my file.",
        "I am extremely angry and disappointed with this awful service.",
    ]
    for text in negative_samples:
        result = analyzer.analyze(text)
        assert result.label == SENTIMENT_NEGATIVE, f"Expected Negative for: {text}"
        assert result.compound < -0.05
        assert 0.0 <= result.confidence <= 1.0


def test_neutral_classification(analyzer):
    neutral_samples = [
        "The shipment was dispatched on Tuesday at 4:30 PM via standard ground freight.",
        "The documentation is located in the root directory under docs.",
        "The meeting has been scheduled for tomorrow morning.",
    ]
    for text in neutral_samples:
        result = analyzer.analyze(text)
        assert result.label == SENTIMENT_NEUTRAL, f"Expected Neutral for: {text}"
        assert -0.05 <= result.compound <= 0.05
        assert 0.0 <= result.confidence <= 1.0


def test_mixed_sentiment_detection(analyzer):
    mixed_samples = [
        "The camera quality is absolutely gorgeous, but the battery life is terrible.",
        "I love the sleek design, however customer support was totally useless.",
    ]
    for text in mixed_samples:
        result = analyzer.analyze(text)
        assert result.is_mixed is True, f"Expected is_mixed=True for: {text}"
        # Confidence score should honestly reflect nuance (diminished compared to pure positive/negative)
        assert result.confidence < 0.95


def test_ambiguous_sentiment_handling(analyzer):
    ambiguous_samples = [
        "It is what it is.",
        "Well, maybe.",
        "Hmm, okay.",
    ]
    for text in ambiguous_samples:
        result = analyzer.analyze(text)
        # Should be neutral or flagged as ambiguous
        assert result.label == SENTIMENT_NEUTRAL or result.is_ambiguous
        # Honest confidence: should not claim high confidence
        assert result.confidence <= 0.85


# -----------------------------------------------------------------------------
# 2. Edge Cases, Empty Input & Error Handling
# -----------------------------------------------------------------------------

def test_empty_and_whitespace_input(analyzer):
    for empty_input in ["", "   ", "\n\t", None]:
        result = analyzer.analyze(empty_input)
        assert result.label == SENTIMENT_NEUTRAL
        assert result.compound == 0.0
        assert result.is_ambiguous is True
        assert "Empty" in result.explanation or "whitespace" in result.explanation


def test_crisis_safety_override(analyzer):
    crisis_text = "I feel completely hopeless and want to end my life."
    result = analyzer.analyze(crisis_text)
    assert result.is_crisis is True

    # Check that safety response overrides normal banter
    adapted = analyzer.adapt_response("Here is the answer to your question.", result)
    assert "988" in adapted
    assert "emergency" in adapted.lower() or "crisis" in adapted.lower()


# -----------------------------------------------------------------------------
# 3. Response Style Adaptation
# -----------------------------------------------------------------------------

def test_tone_adaptation_positive(analyzer):
    pos_res = analyzer.analyze("I am really happy with this great result!")
    base_text = "Your account balance is $250."
    adapted = analyzer.adapt_response(base_text, pos_res)
    # Must preserve factual content
    assert base_text in adapted
    # Must include positive encouragement
    assert "Great" in adapted or "celebrate" in adapted


def test_tone_adaptation_negative(analyzer):
    neg_res = analyzer.analyze("This app is frustrating and broken.")
    base_text = "Please restart the server to apply the patch."
    adapted = analyzer.adapt_response(base_text, neg_res)
    # Must preserve factual content
    assert base_text in adapted
    # Must be empathetic and solution-oriented
    assert "frustrating" in adapted.lower() or "help" in adapted.lower()


# -----------------------------------------------------------------------------
# 4. Zero-API-Key Retrieval Engine Tests
# -----------------------------------------------------------------------------

def test_retrieval_refund_policy(retrieval_engine):
    query = "How do I get a refund or return my package?"
    matches = retrieval_engine.retrieve(query, top_k=2)
    assert len(matches) > 0
    top_match = matches[0]
    assert "refund" in top_match["title"].lower() or "refund" in top_match["content"].lower()
    assert top_match["score"] > 0.12


def test_retrieval_answer_generation(retrieval_engine, analyzer):
    query = "What is the return policy?"
    sentiment = analyzer.analyze("I am asking about the return policy.")
    answer, sources, mode = retrieval_engine.answer_query(query, sentiment)

    assert "Free Retrieval Mode" in mode
    assert len(sources) > 0
    assert "30-day" in answer or "refund" in answer.lower()


def test_retrieval_custom_document_ingestion(retrieval_engine):
    doc_id = retrieval_engine.add_document(
        title="Antigravity Quantum Engine Specifications",
        content="Antigravity Quantum Engine uses superconducting flux qubits operating at 15 millikelvin.",
        source="Custom Hardware Manual",
        category="Quantum Tech",
    )
    assert doc_id.startswith("kb_custom_")

    matches = retrieval_engine.retrieve("superconducting flux qubits quantum", top_k=1)
    assert len(matches) == 1
    assert matches[0]["title"] == "Antigravity Quantum Engine Specifications"


def test_retrieval_unmatched_query_fallback(retrieval_engine, analyzer):
    query = "xyz123nonexistentwordabcdefqwert"
    sentiment = analyzer.analyze(query)
    answer, sources, mode = retrieval_engine.answer_query(query, sentiment)
    assert sources == []
    assert "couldn't find a direct match" in answer.lower()


# -----------------------------------------------------------------------------
# 5. Gemini Service Graceful Handling & Mocked Testing
# -----------------------------------------------------------------------------

def test_gemini_service_without_api_key():
    # When no key is set, is_available must be False and it must not raise any exceptions
    service = GeminiService(api_key="")
    assert service.is_available() is False

    analyzer = SentimentAnalyzer()
    sentiment = analyzer.analyze("Hello there")
    answer, sources, mode = service.generate_response("Tell me about shipping", sentiment)
    # Must fallback cleanly to retrieval mode
    assert "Free Retrieval Mode" in mode


@patch("services.gemini_service.GeminiService.is_available", return_value=True)
def test_gemini_service_mocked_generation(mock_is_avail):
    service = GeminiService(api_key="fake-test-key")
    service._sdk_type = "legacy"
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is a mock Gemini generated response adapted to positive tone."
    mock_model.generate_content.return_value = mock_response
    service._client = mock_model

    analyzer = SentimentAnalyzer()
    sentiment = analyzer.analyze("I love this!")
    answer, sources, mode = service.generate_response("How is everything?", sentiment)

    assert "mock Gemini generated response" in answer
    assert "Google Gemini AI" in mode
