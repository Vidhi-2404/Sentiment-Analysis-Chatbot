"""
Free, Zero-API-Key Knowledge Retrieval Engine using TF-IDF and Cosine Similarity.
Provides local document ingestion, semantic search, source attribution, and
sentiment-conditioned answer generation without any external API calls.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import DEFAULT_TOP_K, MIN_RETRIEVAL_SCORE


# Default domain knowledge base documents
DEFAULT_KNOWLEDGE_BASE: List[Dict[str, str]] = [
    {
        "id": "kb_nlp_01",
        "title": "What is Sentiment Analysis & How VADER Works",
        "category": "NLP Concepts",
        "content": (
            "Sentiment analysis is an NLP technique used to determine whether data is positive, "
            "negative, or neutral. VADER (Valence Aware Dictionary and sEntiment Reasoner) is a "
            "rule-based sentiment analysis engine specifically tuned to sentiments expressed in "
            "conversational text, social media, emojis, and punctuation. It calculates pos, neu, "
            "neg scores and a normalized compound score ranging from -1.0 (most extreme negative) "
            "to +1.0 (most extreme positive)."
        ),
        "source": "NLP Knowledge Base: VADER Principles",
    },
    {
        "id": "kb_nlp_02",
        "title": "Interpreting Compound and Confidence Scores",
        "category": "NLP Concepts",
        "content": (
            "A compound score >= +0.05 indicates positive sentiment, <= -0.05 indicates negative "
            "sentiment, and scores between -0.05 and +0.05 denote neutral sentiment. "
            "Confidence is derived by calculating the margin of separation from neutrality "
            "and penalizing conflicting polarities (mixed feedback)."
        ),
        "source": "NLP Knowledge Base: Metric Calibration",
    },
    {
        "id": "kb_support_01",
        "title": "Product Refund & Return Policy",
        "category": "Customer Support",
        "content": (
            "We offer a 30-day money-back guarantee on all standard orders. If you are dissatisfied "
            "with your purchase, you can request a full refund or exchange within 30 days of delivery. "
            "Items must be in original condition. Refunds are processed back to the original payment "
            "method within 3 to 5 business days."
        ),
        "source": "Customer Support Portal: Refunds & Returns FAQ",
    },
    {
        "id": "kb_support_02",
        "title": "Shipping Times, Tracking & Delivery Delays",
        "category": "Customer Support",
        "content": (
            "Standard shipping takes 3-5 business days within the continental US, and 7-14 days internationally. "
            "Once your order ships, an automated tracking email is dispatched. If your package is delayed "
            "more than 48 hours past the estimated arrival, our 24/7 support team can expedite a replacement."
        ),
        "source": "Customer Support Portal: Shipping Logistics",
    },
    {
        "id": "kb_tech_01",
        "title": "Deploying Streamlit Apps to Streamlit Community Cloud",
        "category": "Technical Guide",
        "content": (
            "To deploy a Python Streamlit app to Streamlit Community Cloud: push your code to a public or "
            "private GitHub repository containing app.py and requirements.txt. Visit share.streamlit.io, "
            "connect your GitHub account, choose the repository and branch, select app.py as the entry point, "
            "and click Deploy. No API key is required if using the local TF-IDF retrieval engine."
        ),
        "source": "Developer Docs: Streamlit Community Cloud Guide",
    },
    {
        "id": "kb_tech_02",
        "title": "Handling API Rate Limits and Fallbacks",
        "category": "Technical Guide",
        "content": (
            "When integrating external LLM APIs (like Google Gemini), always implement graceful degradation. "
            "If the API key is missing or quota is exhausted (HTTP 429), the application should automatically "
            "switch to the local TF-IDF retrieval engine to ensure continuous, zero-downtime availability."
        ),
        "source": "Developer Docs: Resilience & Fallback Architectures",
    },
    {
        "id": "kb_wellness_01",
        "title": "Stress Management & Mental Well-Being Best Practices",
        "category": "Well-Being & Safety",
        "content": (
            "If you are feeling overwhelmed, taking structured breaks, practicing deep diaphragmatic breathing "
            "(4-7-8 method), and reaching out to trusted colleagues or counselors can significantly lower acute stress. "
            "Remember that conversational chatbots provide informational support and are not a replacement for professional clinical care."
        ),
        "source": "Health & Safety Guidelines: Well-Being Protocols",
    },
]


class RetrievalEngine:
    """
    TF-IDF Vectorizer based retrieval engine for zero-API-key question answering.
    """

    def __init__(self, initial_docs: Optional[List[Dict[str, str]]] = None):
        self.documents: List[Dict[str, str]] = list(initial_docs or DEFAULT_KNOWLEDGE_BASE)
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._build_index()

    def _build_index(self):
        """Fit TF-IDF vectorizer on existing knowledge base documents."""
        if not self.documents:
            self.vectorizer = None
            self.tfidf_matrix = None
            return

        corpus = [f"{doc.get('title', '')} {doc.get('content', '')}" for doc in self.documents]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def add_document(self, title: str, content: str, source: str = "User Uploaded", category: str = "Custom") -> str:
        """Add a custom document to the knowledge base and re-index."""
        doc_id = f"kb_custom_{len(self.documents) + 1}"
        new_doc = {
            "id": doc_id,
            "title": title.strip() or f"Document #{len(self.documents) + 1}",
            "category": category.strip() or "Custom",
            "content": content.strip(),
            "source": source.strip() or "Custom Ingestion",
        }
        self.documents.append(new_doc)
        self._build_index()
        return doc_id

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Search for top_k relevant documents matching the query.
        Returns list of matched documents with similarity scores.
        """
        if not query or not query.strip() or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        cleaned_query = query.strip()
        query_vec = self.vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Sort descending by score
        top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices[:top_k]:
            score = float(similarities[idx])
            if score >= MIN_RETRIEVAL_SCORE:
                doc = self.documents[idx].copy()
                doc["score"] = round(score, 4)
                results.append(doc)

        return results

    def answer_query(self, query: str, sentiment_result: Any) -> Tuple[str, List[Dict[str, Any]], str]:
        """
        Generate answer from knowledge base conditioned on sentiment analysis.
        Returns: (answer_text, sources, mode_label)
        """
        matches = self.retrieve(query, top_k=DEFAULT_TOP_K)
        mode_label = "Free Retrieval Mode (TF-IDF Knowledge Base)"

        if matches:
            best_match = matches[0]
            base_answer = (
                f"{best_match['content']}\n\n"
                f"*(Matched topic: **{best_match['title']}** with {best_match['score']:.0%} relevance)*"
            )
            sources = [
                {
                    "title": m["title"],
                    "source": m["source"],
                    "category": m["category"],
                    "score": m["score"],
                }
                for m in matches
            ]
        else:
            # Fallback answer when no direct knowledge document matches the query
            base_answer = (
                f"I couldn't find a direct match in our active knowledge base for: \"{query}\". "
                "However, I have logged your sentiment and query. You can add custom documents in the "
                "sidebar knowledge manager, or ask about our core topics: Sentiment Analysis, "
                "Refund Policies, Shipping, or Streamlit Deployment."
            )
            sources = []

        # Apply sentiment-adapted styling
        from services.sentiment_service import SentimentAnalyzer
        adapted_answer = SentimentAnalyzer().adapt_response(base_answer, sentiment_result)

        return adapted_answer, sources, mode_label
