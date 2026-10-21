# 🧠 Sentiment Analysis Chatbot & Dashboard

A production-ready, standalone **Sentiment Analysis Chatbot & Analytics Dashboard** built with **Python 3.10/3.11**, **Streamlit**, **VADER**, **scikit-learn (TF-IDF)**, and **Plotly**.

This application runs **100% free on Streamlit Community Cloud without requiring any API key**, while supporting optional Google Gemini AI conversational generation when an API key is provided.

---

## 🌟 Key Features

### 1. Real-Time Sentiment Classification
- Classifies user messages into **Positive**, **Neutral**, and **Negative**.
- Powered by **VADER** (*Valence Aware Dictionary and sEntiment Reasoner*), optimized for conversational language, emoticons, emojis, capitalization (`GREAT` vs `great`), and punctuation emphasis (`good!` vs `good!!!`).
- Zero heavy weight downloads: 0 MB checkpoint download on startup ensures instantaneous cold starts and no Streamlit Cloud out-of-memory (OOM) errors.

### 2. Calibrated Confidence Scoring & Nuance Detection
- Generates a bounded confidence metric ($0.0$ to $1.0$) based on distance from neutrality and polarity dominance.
- **Mixed Sentiment Detection**: Automatically identifies conflicting clauses (e.g., *"The camera is gorgeous, but the battery life is terrible"*) and applies an honest confidence penalty.
- **Ambiguity & Hedging Recognition**: Identifies hedging markers (*"maybe"*, *"perhaps"*, *"it is what it is"*) without falsely claiming high confidence.

### 3. Adaptive Response Tone
- **Positive Sentiment**: Warm, encouraging, and celebratory tone.
- **Neutral Sentiment**: Objective, clear, and informative tone.
- **Negative Sentiment**: Empathetic, calm, validating, and solution-oriented tone.
- **Intent & Factuality Preservation**: Strict safety guardrail ensures tone adjustments never alter underlying factual answers or safety advice.
- **Crisis Safety Protocol**: Automatically intercepts acute distress keywords with 24/7 crisis hotlines (988 Lifeline).

### 4. Zero-API-Key Retrieval Mode (TF-IDF Knowledge Base)
- Default out-of-the-box mode requires **zero API keys**.
- Uses `scikit-learn` TF-IDF vectorization and cosine similarity to index and search domain knowledge documents.
- Clearly displays source attribution badges and relevance percentages.
- Interactive Knowledge Base manager allows live ingestion of custom documents right in the sidebar.

### 5. Optional Google Gemini AI Integration
- If `GEMINI_API_KEY` is configured via environment variable, `.env`, Streamlit secrets, or user sidebar input, unlocks full generative AI conversational responses.
- If no key is supplied, **the app never crashes**; it cleanly and automatically defaults to the Free Retrieval Engine.

### 6. Interactive Analytics & Telemetry Dashboard
- Metrics overview: Total messages analyzed, Positive %, Neutral %, Negative %, and Average Confidence.
- **Interactive Plotly Donut Chart**: Sentiment distribution breakdown.
- **Interactive Plotly Trajectory Chart**: Turn-by-turn compound valence and confidence timeline.
- **Filterable Data Table**: Message-by-message audit trail.
- **One-Click Synthetic Benchmark Loader**: Instantly populates the dashboard with 15 diverse benchmark records for demonstration.
- **CSV Export**: Download anonymized telemetry data with no permanent personal data storage.

---

## 📂 Project Architecture

```
Sentiment Analysis Chatbot/
├── app.py                      # Streamlit application entry point (UI & Dashboard)
├── config.py                   # Global constants, thresholds, tone templates, safety rules
├── requirements.txt            # Lightweight, pinned production dependencies
├── .env.example                # Template for environment configuration
├── .gitignore                  # Git ignore rules for virtualenvs and secrets
├── README.md                   # Comprehensive documentation and deployment guide
├── services/
│   ├── sentiment_service.py    # VADER analyzer, confidence scoring & tone adaptation
│   ├── retrieval_service.py    # Free zero-API-key TF-IDF knowledge retrieval engine
│   └── gemini_service.py       # Safe optional Gemini LLM integration with fallback
└── tests/
    └── test_sentiment.py       # Pytest unit tests for classification, retrieval & safety
```

---

## 🚀 Local Installation & Setup

### 1. Clone or Open the Repository
```bash
git clone https://github.com/<your-username>/sentiment-analysis-chatbot.git
cd sentiment-analysis-chatbot
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Configure Gemini API Key
Copy `.env.example` to `.env` if you want to use Google Gemini:
```bash
cp .env.example .env
```
Add your key inside `.env`:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
```
> **Note:** This step is completely optional! The app runs 100% fully functional in Free Retrieval Mode without any key.

### 5. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing

Run the automated test suite with `pytest`:
```bash
pytest tests/test_sentiment.py -v
```

### Evaluated Test Benchmark Results
| Test Category | Description | Status |
|---|---|---|
| **Positive Classification** | Verifies compound score $> +0.05$ on enthusiastic feedback | ✅ PASS |
| **Negative Classification** | Verifies compound score $< -0.05$ on critical support tickets | ✅ PASS |
| **Neutral Classification** | Verifies $|compound| \le 0.05$ on factual logistics statements | ✅ PASS |
| **Mixed Sentiment** | Detects conflicting clauses (`pos >= 0.12` and `neg >= 0.12`) | ✅ PASS |
| **Ambiguity & Hedging** | Detects phrases like *"maybe"*, *"perhaps"*, *"it is what it is"* | ✅ PASS |
| **Empty Input Handling** | Gracefully handles empty and whitespace strings | ✅ PASS |
| **Crisis Safety Override** | Triggers 988 emergency lifeline override on distress keywords | ✅ PASS |
| **Tone Adaptation** | Validates factual preservation with empathetic/encouraging tones | ✅ PASS |
| **TF-IDF Retrieval** | Verifies document matching and source attribution | ✅ PASS |
| **Document Ingestion** | Indexes new user-submitted documents dynamically | ✅ PASS |
| **Zero-Key Gemini Safety** | Verifies zero crashes when `GEMINI_API_KEY` is missing | ✅ PASS |

---

## ☁️ Deployment to Streamlit Community Cloud (Zero API Key)

Deploying this app requires **no credit card and zero API keys**.

### Step 1: Create a GitHub Repository
1. Go to [GitHub.com/new](https://github.com/new).
2. Repository name: `sentiment-analysis-chatbot`.
3. Set visibility to **Public**.
4. In your local terminal:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Sentiment Analysis Chatbot & Dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/sentiment-analysis-chatbot.git
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"**.
3. Select:
   - **Repository:** `<your-username>/sentiment-analysis-chatbot`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy!"**.
5. Your public URL will be ready in under 60 seconds (e.g., `https://sentiment-analysis-chatbot-<hash>.streamlit.app`).

### Step 3: (Optional) Adding Gemini API Key on Streamlit Cloud
If you decide to enable Gemini AI on Streamlit Cloud:
1. In your app's dashboard on `share.streamlit.io`, go to **Settings** -> **Secrets**.
2. Add:
   ```toml
   GEMINI_API_KEY = "AIzaSy..."
   ```
3. Save. The app will immediately recognize the key and enable Gemini Mode.

---

## 🔬 Model Strengths & Documented Limitations

### Strengths
1. **Zero Cold-Start Latency**: VADER is bundled inside its Python package; there are no gigabyte-scale model downloads at startup.
2. **Conversational Optimization**: Accurately gauges emoticons (`:)`, `:(`), punctuation amplification (`awesome!!!`), and capitalized emphasis (`SO BAD`).
3. **Privacy & Ephemerality**: User chat inputs are processed in-memory during the active browser session and are never permanently stored on external servers.

### Documented Limitations
1. **Sarcasm / Irony**: Rule-based lexical models cannot reliably decode complex irony (e.g., *"Oh perfect, my hard drive just died!"*).
2. **Domain-Specific Inversions**: In medical contexts, *"test results came back negative"* is positive news, but lexical dictionaries evaluate *"negative"* as unfavorable.
3. **Syntactic Nesting**: Intricate double-negation or multi-paragraph contrasting arguments may attenuate compound confidence.

---

## 📄 License
MIT License. Built for internship submission and public demonstration.
