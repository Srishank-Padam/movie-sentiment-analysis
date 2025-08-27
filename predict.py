import argparse, re, joblib
from pathlib import Path

def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return s.strip()

def load_artifacts(art_dir="artifacts"):
    model = joblib.load(Path(art_dir) / "model.pkl")
    vectorizer = joblib.load(Path(art_dir) / "vectorizer.pkl")
    return model, vectorizer

def predict_once(text: str, model, vectorizer):
    text = clean_text(text)
    X = vectorizer.transform([text])
    proba = getattr(model, "predict_proba", None)
    if proba:
        p = model.predict_proba(X)[0][1]  # prob of positive
        label = "Positive 😊" if p >= 0.5 else "Negative 😞"
        return label, float(p)
    else:
        pred = model.predict(X)[0]
        return ("Positive 😊" if pred == 1 else "Negative 😞", None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="Review text to classify")
    parser.add_argument("--art_dir", default="artifacts")
    args = parser.parse_args()

    model, vectorizer = load_artifacts(args.art_dir)

    if args.text:
        label, p = predict_once(args.text, model, vectorizer)
        if p is None:
            print(f"Prediction: {label}")
        else:
            print(f"Prediction: {label}  (confidence={p:.3f})")
    else:
        # Interactive mode
        print("Type a review and press Enter (empty line to quit):")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            label, p = predict_once(line, model, vectorizer)
            if p is None:
                print(f"→ {label}")
            else:
                print(f"→ {label}  (confidence={p:.3f})")
