"""
Sentiment Analysis Chatbot & Dashboard
Streamlit Application Entry Point
Supports 100% Free Zero-API-Key Retrieval Mode and Optional Google Gemini Mode.
"""

import os
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Load local environment if present
load_dotenv()

from config import (
    APP_TITLE,
    APP_SUBTITLE,
    APP_VERSION,
    SENTIMENT_POSITIVE,
    SENTIMENT_NEUTRAL,
    SENTIMENT_NEGATIVE,
    SENTIMENT_COLORS,
    SENTIMENT_EMOJIS,
    TONE_DIRECTIVES,
    SAFETY_DISCLAIMER,
)
from services.sentiment_service import SentimentAnalyzer, SentimentResult
from services.retrieval_service import RetrievalEngine, DEFAULT_KNOWLEDGE_BASE
from services.gemini_service import GeminiService


# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern aesthetics, badges, and responsive bubbles
st.markdown(
    """
    <style>
    /* Global font & smooth transitions */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero Banner styling */
    .hero-container {
        padding: 1.2rem 1.6rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 0;
    }

    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.22rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-right: 0.35rem;
    }
    .badge-positive {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .badge-neutral {
        background-color: rgba(107, 114, 128, 0.15);
        color: #9ca3af;
        border: 1px solid rgba(107, 114, 128, 0.35);
    }
    .badge-negative {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .badge-mode {
        background-color: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-tone {
        background-color: rgba(168, 85, 247, 0.12);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    .badge-score {
        background-color: rgba(255, 255, 255, 0.06);
        color: #cbd5e1;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }

    /* Metric Card styling */
    .metric-card {
        padding: 1.1rem;
        border-radius: 12px;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px -3px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Source citations */
    .source-box {
        padding: 0.6rem 0.85rem;
        margin-top: 0.5rem;
        background: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #38bdf8;
        border-radius: 4px;
        font-size: 0.82rem;
        color: #cbd5e1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrieval_engine" not in st.session_state:
    st.session_state.retrieval_engine = RetrievalEngine()

if "sentiment_analyzer" not in st.session_state:
    st.session_state.sentiment_analyzer = SentimentAnalyzer()

if "analytics_log" not in st.session_state:
    # Initialize with a few realistic synthetic benchmark queries for immediate visual richness
    st.session_state.analytics_log = []

if "gemini_api_key" not in st.session_state:
    # Check environment variable first
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def get_sentiment_badge_html(sentiment: SentimentResult) -> str:
    """Generate HTML pill badges for sentiment, confidence, and flags."""
    badge_cls = {
        SENTIMENT_POSITIVE: "badge-positive",
        SENTIMENT_NEUTRAL: "badge-neutral",
        SENTIMENT_NEGATIVE: "badge-negative",
    }.get(sentiment.label, "badge-neutral")

    emoji = SENTIMENT_EMOJIS.get(sentiment.label, "💬")
    conf_pct = f"{sentiment.confidence:.0%}"
    html = f'<span class="badge {badge_cls}">{emoji} {sentiment.label} ({conf_pct} conf)</span>'
    html += f'<span class="badge badge-score">Valence: {sentiment.compound:+.2f}</span>'

    if sentiment.is_mixed:
        html += '<span class="badge badge-score">⚖️ Mixed Signals</span>'
    if sentiment.is_ambiguous:
        html += '<span class="badge badge-score">❓ Ambiguous</span>'
    if sentiment.is_crisis:
        html += '<span class="badge badge-negative">⚠️ Crisis Safety Triggered</span>'

    return html


def load_synthetic_demo_data():
    """Populate analytics and chat with diverse realistic benchmark samples."""
    samples = [
        ("I am totally thrilled with this outstanding application! 🚀", "Customer Feedback"),
        ("The product is completely broken and failed right before my presentation.", "Support Ticket"),
        ("The package was delivered on Wednesday at 2:15 PM.", "Logistics Query"),
        ("The user interface is gorgeous, but the server response latency is terrible.", "Product Review"),
        ("Well, I suppose it is fine.", "General Observation"),
        ("Antigravity paired with Streamlit makes full-stack NLP engineering remarkably fast!", "Developer Log"),
        ("Customer support refused to honor the 30-day money-back guarantee.", "Complaint"),
        ("Water boils at 100 degrees Celsius under standard atmospheric pressure.", "Factual FAQ"),
        ("Great job resolving my shipping problem so quickly! Highly recommended.", "Testimonial"),
        ("I feel frustrated because the documentation lacks clear configuration examples.", "Feedback"),
        ("The payment went through without any verification prompt.", "Security Question"),
        ("I love the minimalist dark mode aesthetic!", "UI Evaluation"),
        ("It is what it is.", "Hedging Message"),
        ("The app crashes whenever I attempt to upload a CSV larger than 50MB.", "Bug Report"),
        ("Could you explain how VADER calculates compound sentiment scores?", "Technical Inquiry"),
    ]

    new_logs = []
    for text, category in samples:
        res = st.session_state.sentiment_analyzer.analyze(text)
        new_logs.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": text,
            "category": category,
            "label": res.label,
            "compound": res.compound,
            "confidence": res.confidence,
            "pos": res.detailed_scores.get("pos", 0.0),
            "neu": res.detailed_scores.get("neu", 0.0),
            "neg": res.detailed_scores.get("neg", 0.0),
            "is_mixed": res.is_mixed,
            "is_ambiguous": res.is_ambiguous,
        })
    st.session_state.analytics_log.extend(new_logs)


# -----------------------------------------------------------------------------
# Sidebar: Engine Configuration, KB Manager, Benchmarks & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/streamlit/brand/main/streamlit-mark-color.png",
        width=48,
    )
    st.title("Settings & Engine")
    st.caption("Sentiment Analysis Chatbot v" + APP_VERSION)

    st.markdown("---")
    st.subheader("⚙️ Operating Mode")

    # Mode selection
    mode_options = [
        "Free Retrieval Mode (Zero API Key)",
        "Google Gemini AI (Optional API Key)",
    ]
    selected_mode = st.radio(
        "Response Generator:",
        mode_options,
        index=0,
        help="Free Retrieval Mode runs 100% locally with TF-IDF without any external API or secret.",
    )

    gemini_key_input = ""
    is_gemini_active = False

    if selected_mode == "Google Gemini AI (Optional API Key)":
        gemini_key_input = st.text_input(
            "Gemini API Key:",
            value=st.session_state.gemini_api_key,
            type="password",
            placeholder="AIzaSy...",
            help="Optional: Enter key to enable Gemini AI mode. If left blank, automatically uses Free Retrieval Mode.",
        )
        if gemini_key_input:
            st.session_state.gemini_api_key = gemini_key_input
            gemini_svc = GeminiService(api_key=gemini_key_input)
            if gemini_svc.is_available():
                st.success("🟢 Gemini API Connected")
                is_gemini_active = True
            else:
                st.warning("⚠️ Key entered but initialization failed. Falling back to Free Retrieval.")
        else:
            st.info("ℹ️ No key entered. Using Free Retrieval Mode.")
    else:
        st.success("🟢 100% Free Local Mode Active (Zero API Key)")

    st.markdown("---")
    st.subheader("📚 Knowledge Base")
    doc_count = len(st.session_state.retrieval_engine.documents)
    st.caption(f"Active Indexed Documents: **{doc_count}**")

    with st.expander("📖 View / Ingest Documents"):
        st.markdown("**Preloaded Topics:**")
        for d in st.session_state.retrieval_engine.documents[:4]:
            st.markdown(f"- *{d['title']}* (`{d['category']}`)")

        st.markdown("---")
        st.markdown("**Add Custom Knowledge:**")
        with st.form("add_doc_form", clear_on_submit=True):
            new_title = st.text_input("Title", placeholder="e.g. Return Address & Hours")
            new_category = st.text_input("Category", placeholder="e.g. Operations")
            new_content = st.text_area("Content", placeholder="Enter document text or policy information...")
            submitted = st.form_submit_button("Ingest Document")
            if submitted and new_title and new_content:
                new_id = st.session_state.retrieval_engine.add_document(
                    title=new_title,
                    content=new_content,
                    category=new_category or "Custom",
                    source="User UI Ingestion",
                )
                st.success(f"Indexed as `{new_id}`!")
                st.rerun()

    st.markdown("---")
    st.subheader("🧪 Quick-Test Benchmarks")
    st.caption("Click to test classification and adaptive tone:")

    benchmark_col1, benchmark_col2 = st.columns(2)
    sample_to_inject = None

    with benchmark_col1:
        if st.button("😊 Positive", use_container_width=True):
            sample_to_inject = "I absolutely love how responsive and intuitive this tool is!"
        if st.button("😡 Negative", use_container_width=True):
            sample_to_inject = "This product completely broke my workflow and is super frustrating."
        if st.button("❓ Ambiguous", use_container_width=True):
            sample_to_inject = "Well, I guess so."

    with benchmark_col2:
        if st.button("😐 Neutral", use_container_width=True):
            sample_to_inject = "What is the return policy for international shipping?"
        if st.button("⚖️ Mixed", use_container_width=True):
            sample_to_inject = "The design is gorgeous, but the performance is unacceptably slow."

    st.markdown("---")
    st.subheader("🧹 Actions")
    if st.button("Clear Chat Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("Load 15 Synthetic Logs", use_container_width=True):
        load_synthetic_demo_data()
        st.success("Loaded 15 synthetic benchmark records into Dashboard!")
        st.rerun()


# -----------------------------------------------------------------------------
# Main Header Banner
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-container">
        <div class="hero-title">{APP_TITLE}</div>
        <p class="hero-subtitle">{APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Primary Tabs: Chatbot vs Analytics Dashboard
# -----------------------------------------------------------------------------
tab_chat, tab_dashboard, tab_docs = st.tabs([
    "💬 Sentiment-Aware Chatbot",
    "📊 Analytics & Dashboard",
    "ℹ️ Model Transparency & Documentation",
])


# =============================================================================
# TAB 1: Chatbot Interface
# =============================================================================
with tab_chat:
    # Live status bar
    status_cols = st.columns([2, 1, 1])
    with status_cols[0]:
        active_mode_desc = (
            "🤖 Google Gemini AI Mode" if is_gemini_active
            else "🔍 Free Retrieval Mode (TF-IDF Knowledge Base - No API Key Required)"
        )
        st.caption(f"**Current Engine:** {active_mode_desc}")
    with status_cols[1]:
        st.caption(f"**Model:** VADER Lexicon Analyzer")
    with status_cols[2]:
        msg_count = len(st.session_state.messages)
        st.caption(f"**Turns in History:** {msg_count // 2}")

    st.markdown("---")

    # Render Chat History
    if not st.session_state.messages:
        st.info(
            "👋 **Welcome!** Type a question, comment, or review below to see real-time sentiment "
            "classification, confidence scoring, and adaptive responses. "
            "Or click any benchmark in the sidebar!"
        )

    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            with st.chat_message("user"):
                st.write(content)
                # Show sentiment telemetry badge bar
                if "sentiment" in msg and msg["sentiment"]:
                    s_res = msg["sentiment"]
                    badge_html = get_sentiment_badge_html(s_res)
                    st.markdown(badge_html, unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.write(content)

                # Metadata chips
                mode_label = msg.get("mode", "Free Retrieval Engine")
                tone_badge = msg.get("tone_badge", "Informative")
                st.markdown(
                    f'<span class="badge badge-mode">⚙️ {mode_label}</span>'
                    f'<span class="badge badge-tone">🎭 Tone: {tone_badge}</span>',
                    unsafe_allow_html=True,
                )

                # Show sources if any were retrieved
                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📚 Retrieved Context Sources ({len(sources)} matched)"):
                        for src in sources:
                            score_pct = f"{src.get('score', 0):.0%}"
                            st.markdown(
                                f"""
                                <div class="source-box">
                                    <strong>{src.get('title')}</strong> — <em>Relevance: {score_pct}</em><br/>
                                    <small>Category: {src.get('category')} | Source: {src.get('source')}</small>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Detailed Sentiment Breakdown Expander
                if "sentiment" in msg and msg["sentiment"]:
                    s_res = msg["sentiment"]
                    with st.expander("🔬 Sentiment Analysis Diagnostic Details"):
                        d_cols = st.columns(4)
                        with d_cols[0]:
                            st.metric("Compound Score", f"{s_res.compound:+.3f}")
                        with d_cols[1]:
                            st.metric("Positivity", f"{s_res.detailed_scores.get('pos', 0):.1%}")
                        with d_cols[2]:
                            st.metric("Neutrality", f"{s_res.detailed_scores.get('neu', 0):.1%}")
                        with d_cols[3]:
                            st.metric("Negativity", f"{s_res.detailed_scores.get('neg', 0):.1%}")
                        st.caption(f"**Classification Rationale:** {s_res.explanation}")

    # Process Input from Chat or Benchmark Button
    user_prompt = st.chat_input("Type your message, ask a question, or express your thoughts...")
    if sample_to_inject:
        user_prompt = sample_to_inject

    if user_prompt:
        if not user_prompt.strip():
            st.warning("⚠️ Please enter a non-empty message.")
        else:
            # 1. Analyze Sentiment
            sentiment_analyzer = st.session_state.sentiment_analyzer
            s_result = sentiment_analyzer.analyze(user_prompt)

            # 2. Append user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_prompt,
                "sentiment": s_result,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

            # 3. Log to analytics
            st.session_state.analytics_log.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": user_prompt,
                "category": "Live Chat",
                "label": s_result.label,
                "compound": s_result.compound,
                "confidence": s_result.confidence,
                "pos": s_result.detailed_scores.get("pos", 0.0),
                "neu": s_result.detailed_scores.get("neu", 0.0),
                "neg": s_result.detailed_scores.get("neg", 0.0),
                "is_mixed": s_result.is_mixed,
                "is_ambiguous": s_result.is_ambiguous,
            })

            # 4. Generate response
            with st.spinner("Analyzing sentiment and formulating adaptive response..."):
                if is_gemini_active:
                    gemini_svc = GeminiService(api_key=st.session_state.gemini_api_key)
                    resp_text, sources, mode_used = gemini_svc.generate_response(
                        user_prompt,
                        s_result,
                        st.session_state.retrieval_engine,
                    )
                else:
                    resp_text, sources, mode_used = st.session_state.retrieval_engine.answer_query(
                        user_prompt,
                        s_result,
                    )

                tone_badge = (
                    s_result.tone_directive.get("badge", "Informative")
                    if s_result.tone_directive
                    else TONE_DIRECTIVES[s_result.label]["badge"]
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": resp_text,
                    "sentiment": s_result,
                    "sources": sources,
                    "mode": mode_used,
                    "tone_badge": tone_badge,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

            st.rerun()


# =============================================================================
# TAB 2: Analytics & Sentiment Dashboard
# =============================================================================
with tab_dashboard:
    st.subheader("📈 Sentiment Telemetry & Analytics Dashboard")
    st.caption("Aggregated analytics from conversation turns and synthetic benchmarks.")

    log_data = st.session_state.analytics_log

    if not log_data:
        st.info("No messages have been analyzed yet.")
        if st.button("📊 Populate with 15 Synthetic Benchmarks Now"):
            load_synthetic_demo_data()
            st.rerun()
    else:
        df = pd.DataFrame(log_data)
        total_msgs = len(df)
        pos_count = int((df["label"] == SENTIMENT_POSITIVE).sum())
        neu_count = int((df["label"] == SENTIMENT_NEUTRAL).sum())
        neg_count = int((df["label"] == SENTIMENT_NEGATIVE).sum())
        mixed_count = int(df["is_mixed"].sum())
        avg_conf = float(df["confidence"].mean())
        avg_compound = float(df["compound"].mean())

        # Metric summary cards
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #38bdf8;">{total_msgs}</div>
                    <div class="metric-label">Total Analyzed</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #10b981;">{pos_count} <small style="font-size:1rem;">({pos_count/total_msgs:.0%})</small></div>
                    <div class="metric-label">Positive 😊</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #9ca3af;">{neu_count} <small style="font-size:1rem;">({neu_count/total_msgs:.0%})</small></div>
                    <div class="metric-label">Neutral 😐</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #ef4444;">{neg_count} <small style="font-size:1rem;">({neg_count/total_msgs:.0%})</small></div>
                    <div class="metric-label">Negative 😟</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col5:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #c084fc;">{avg_conf:.1%}</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br/>", unsafe_allow_html=True)

        # Plotly Charts Row
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("##### 🍩 Sentiment Distribution")
            label_counts = df["label"].value_counts().reset_index()
            label_counts.columns = ["Sentiment", "Count"]

            fig_donut = px.pie(
                label_counts,
                values="Count",
                names="Sentiment",
                hole=0.55,
                color="Sentiment",
                color_discrete_map=SENTIMENT_COLORS,
            )
            fig_donut.update_traces(
                textposition="inside",
                textinfo="percent+label",
                marker=dict(line=dict(color="#0f172a", width=2)),
            )
            fig_donut.update_layout(
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with chart_col2:
            st.markdown("##### 📈 Valence Trajectory & Confidence")
            df["turn_idx"] = range(1, len(df) + 1)

            fig_scatter = go.Figure()
            # Compound line
            fig_scatter.add_trace(go.Scatter(
                x=df["turn_idx"],
                y=df["compound"],
                mode="lines+markers",
                name="Compound Valence",
                line=dict(color="#38bdf8", width=2),
                marker=dict(size=8, color=[SENTIMENT_COLORS.get(l, "#94a3b8") for l in df["label"]]),
            ))
            # Zero baseline
            fig_scatter.add_hline(y=0.0, line_dash="dash", line_color="#475569", annotation_text="Neutral Line")
            fig_scatter.add_hline(y=0.05, line_dash="dot", line_color="rgba(16,185,129,0.5)")
            fig_scatter.add_hline(y=-0.05, line_dash="dot", line_color="rgba(239,68,68,0.5)")

            fig_scatter.update_layout(
                xaxis_title="Turn Index",
                yaxis_title="Valence (-1.0 to +1.0)",
                yaxis_range=[-1.05, 1.05],
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Detailed Breakdown Table
        st.markdown("---")
        st.markdown("##### 📋 Telemetry & Classification Log")

        filter_col1, filter_col2 = st.columns([1, 3])
        with filter_col1:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment:",
                ["All", SENTIMENT_POSITIVE, SENTIMENT_NEUTRAL, SENTIMENT_NEGATIVE],
            )

        filtered_df = df if sentiment_filter == "All" else df[df["label"] == sentiment_filter]

        display_df = filtered_df[[
            "timestamp", "query", "label", "compound", "confidence", "is_mixed", "is_ambiguous"
        ]].copy()
        display_df.rename(columns={
            "timestamp": "Timestamp",
            "query": "Message Text",
            "label": "Sentiment",
            "compound": "Valence",
            "confidence": "Confidence",
            "is_mixed": "Mixed?",
            "is_ambiguous": "Ambiguous?",
        }, inplace=True)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        # Export CSV Button
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Telemetry Data (CSV)",
            data=csv_data,
            file_name=f"sentiment_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


# =============================================================================
# TAB 3: Model Transparency, Limitations & Documentation
# =============================================================================
with tab_docs:
    st.subheader("ℹ️ Model Architecture, Strengths & Documented Limitations")

    st.markdown(
        """
        ### 1. Selected Sentiment Model: VADER (Valence Aware Dictionary and sEntiment Reasoner)
        - **Why selected**:
          - Optimized specifically for social and conversational language, emoticons, emojis, capitalization (`GREAT` vs `great`), and punctuation emphasis (`good!` vs `good!!!`).
          - Lightweight and self-contained: **0 MB checkpoint downloads** on app startup, ensuring ultra-fast cold starts on Streamlit Community Cloud and zero memory-limit termination.
          - 100% deterministic, inspectable, and explainable rule-based lexicon.

        ### 2. Calibrated Confidence Scoring
        - Standard sentiment models often produce raw compound scores between -1.0 and +1.0.
        - In this app, a **normalized confidence metric [0.0, 1.0]** is derived by:
          1. Measuring the distance of the compound score from neutrality thresholds ($\pm0.05$).
          2. Factoring the dominant polarity proportion (`pos` or `neg`).
          3. Applying an explicit penalty ($0.82\\times$) when contradictory positive and negative clauses coexist (**Mixed Sentiment**).
          4. Tagging ambiguous or hedging phrases (*"maybe"*, *"perhaps"*, *"it is what it is"*) with realistic, non-inflated confidence ($0.55 - 0.70$).

        ### 3. Documented Limitations
        - **Sarcasm & Irony**: Lexicon models cannot easily distinguish sarcasm (e.g. *"Oh brilliant, another software crash!"* has positive words but negative intent).
        - **Domain Polarity Inversion**: In medical or cybersecurity contexts, *"test was negative"* or *"zero vulnerabilities"* are positive outcomes, but lexical matchers evaluate *"negative"* and *"zero"* neutrally or critically.
        - **Context Invariance**: Long multi-paragraph essays with subtle transitions can dilute the compound score.

        ### 4. Safety Guardrails & Tone Adaptation
        - **Intent Preservation**: Adapting tone (empthatic, informative, encouraging) strictly preserves underlying facts and never overrides medical or emergency safety.
        - **Crisis Safety Protocol**: Phrases referencing self-harm or acute distress automatically override regular conversational banter, presenting immediate 24/7 hotline resources (988 Lifeline).
        """
    )

    st.info(f"🛡️ **Safety Notice:** {SAFETY_DISCLAIMER}")
