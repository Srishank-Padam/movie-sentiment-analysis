import streamlit as st, joblib, re
from pathlib import Path

ART_DIR = Path("artifacts")
model = joblib.load(ART_DIR / "model.pkl")
vectorizer = joblib.load(ART_DIR / "vectorizer.pkl")

def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return s.strip()

st.title("🎬 Movie Review Sentiment Analyzer")
review = st.text_area("Paste a movie review:")

if st.button("Predict"):
    if not review.strip():
        st.warning("Please paste a review.")
    else:
        X = vectorizer.transform([clean_text(review)])
        proba = getattr(model, "predict_proba", None)
        if proba:
            p = model.predict_proba(X)[0][1]
            label = "✅ Positive" if p >= 0.5 else "❌ Negative"
            st.subheader(f"{label}  (confidence={p:.3f})")
        else:
            pred = model.predict(X)[0]
            label = "✅ Positive" if pred == 1 else "❌ Negative"
            st.subheader(label)
