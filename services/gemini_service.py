"""
Optional Google Gemini Integration Service.
Gracefully handles missing or invalid API keys, providing automatic fallback
to local TF-IDF retrieval mode with zero crashes.
"""

import os
from typing import Optional, List, Dict, Any, Tuple
from services.sentiment_service import SentimentResult
from services.retrieval_service import RetrievalEngine


class GeminiService:
    """
    Manages optional Gemini LLM responses conditioned on sentiment classification.
    Runs 100% safely without crashing when no API key is supplied.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        self._client = None
        self._init_client()

    def _init_client(self):
        """Safely initialize the Gemini client if a key is present."""
        if not self.api_key:
            self._client = None
            return

        try:
            # Try new google-genai SDK first
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            self._sdk_type = "genai"
        except Exception:
            try:
                # Fallback to legacy google.generativeai
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                self._client = legacy_genai.GenerativeModel("gemini-1.5-flash")
                self._sdk_type = "legacy"
            except Exception:
                self._client = None
                self._sdk_type = None

    def is_available(self) -> bool:
        """Check if Gemini API is configured and operational."""
        return bool(self.api_key and self._client is not None)

    def generate_response(
        self,
        query: str,
        sentiment: SentimentResult,
        retrieval_engine: Optional[RetrievalEngine] = None,
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        """
        Generate conversational response informed by sentiment classification.
        Falls back to local TF-IDF retrieval if Gemini is unavailable or errors out.
        """
        # If Gemini is not available, immediately use zero-API-key retrieval
        if not self.is_available():
            fallback_engine = retrieval_engine or RetrievalEngine()
            return fallback_engine.answer_query(query, sentiment)

        # Retrieve relevant background context from knowledge base
        context_snippets = []
        sources = []
        if retrieval_engine:
            matches = retrieval_engine.retrieve(query, top_k=2)
            for m in matches:
                context_snippets.append(f"[{m['title']}]: {m['content']}")
                sources.append({
                    "title": m["title"],
                    "source": m["source"],
                    "category": m["category"],
                    "score": m["score"],
                })

        context_str = "\n".join(context_snippets) if context_snippets else "No specific document matched."

        # Construct sentiment-aware system prompt
        system_instruction = (
            f"You are a helpful, professional Sentiment-Aware AI Assistant.\n"
            f"USER SENTIMENT DETECTED: {sentiment.label} (Compound: {sentiment.compound:.2f}, Confidence: {sentiment.confidence:.0%}).\n"
            f"TONE DIRECTIVE:\n"
            f"- If Positive: Be friendly, encouraging, and enthusiastic.\n"
            f"- If Neutral: Be clear, concise, objective, and informative.\n"
            f"- If Negative: Be empathetic, calm, validating, and solution-oriented.\n\n"
            f"IMPORTANT SAFETY & INTEGRITY RULES:\n"
            f"1. Factual accuracy and safety always take precedence over sentiment tone.\n"
            f"2. Never override medical precautions, legal safety, or emergency warnings.\n"
            f"3. Use the following reference context when answering if relevant:\n"
            f"{context_str}\n"
        )

        prompt = f"{system_instruction}\n\nUSER MESSAGE:\n{query}\n\nYOUR ADAPTIVE RESPONSE:"

        try:
            if self._sdk_type == "genai":
                response = self._client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                text = response.text or "I processed your request, but received an empty response."
            else:
                response = self._client.generate_content(prompt)
                text = response.text or "I processed your request, but received an empty response."

            return text, sources, "Google Gemini AI (LLM Mode)"

        except Exception as e:
            # Graceful error recovery: fall back to TF-IDF retrieval
            fallback_engine = retrieval_engine or RetrievalEngine()
            fallback_ans, fallback_src, _ = fallback_engine.answer_query(query, sentiment)
            notice = (
                f"*(Notice: Gemini API call encountered an error: {type(e).__name__}. "
                f"Automatically fell back to local retrieval mode.)*\n\n"
            )
            return f"{notice}{fallback_ans}", fallback_src, "Free Retrieval Mode (Fallback)"
