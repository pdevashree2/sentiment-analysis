import streamlit as st
import pickle
import re

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.6rem;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .result-card {
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-top: 1.2rem;
        font-size: 1.2rem;
        font-weight: 500;
        text-align: center;
    }

    .positive {
        background-color: #e6f9f0;
        border-left: 6px solid #22c55e;
        color: #15803d;
    }

    .negative {
        background-color: #fff0f0;
        border-left: 6px solid #ef4444;
        color: #b91c1c;
    }

    .metric-box {
        background: #f8f8fc;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e5e5f7;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
    }

    .metric-label {
        font-size: 0.82rem;
        color: #888;
        margin-top: 0.2rem;
    }

    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem;
        color: #1a1a2e;
        margin-top: 2rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #e5e5f7;
        padding-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = pickle.load(open('model.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    return model, tfidf

model, tfidf = load_model()

# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎬 Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Paste any movie review — the model will tell you if it\'s positive or negative.</div>', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
review = st.text_area(
    "Your review",
    placeholder="e.g. An absolutely riveting film with stunning performances...",
    height=160,
    label_visibility="collapsed"
)
analyze = st.button("Analyze Sentiment")
if analyze:
    if review.strip():
        cleaned = clean_text(review)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vectorized)[0]
            confidence = round(max(proba) * 100, 1)
        else:
            confidence = None

        if prediction == 1:
            st.success("😊 Positive Review!")
        else:
            st.error("😞 Negative Review!")

        if confidence is not None:
            st.markdown(f"**Confidence:** {confidence}%")
            st.progress(int(confidence))
    else:
        st.warning("⚠️ Please enter a review before clicking Analyze.")

# ── Prediction ────────────────────────