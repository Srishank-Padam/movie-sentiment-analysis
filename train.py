import argparse, os, re, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return s.strip()

def main(csv_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load
    df = pd.read_csv(csv_path)
    # Expect columns: review, sentiment
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"review","sentiment"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: review, sentiment")

    # 2) Clean + map labels
    df = df.dropna(subset=["review","sentiment"]).copy()
    df["review"] = df["review"].apply(clean_text)
    label_map = {"positive":1, "neg":0, "negative":0, 1:1, 0:0}
    df["label"] = df["sentiment"].map(lambda x: label_map.get(str(x).lower()))
    if df["label"].isna().any():
        raise ValueError("Sentiment values must be positive/negative (or 1/0).")
    X, y = df["review"], df["label"].astype(int)

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Vectorize (word unigrams + bigrams)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1,2),
        max_df=0.9,
        min_df=5,
        max_features=30000,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 5) Train model
    model = LogisticRegression(max_iter=1000, n_jobs=None)
    model.fit(X_train_vec, y_train)

    # 6) Evaluate
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    print(f"\nAccuracy: {acc:.4f}\n")
    print(report)

    # Confusion matrix image
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close(fig)

    # 7) Save artifacts
    joblib.dump(model, os.path.join(out_dir, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(out_dir, "vectorizer.pkl"))
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n{report}")

    print(f"\nSaved: {out_dir}/model.pkl, {out_dir}/vectorizer.pkl, {out_dir}/metrics.txt")

if __name__ == "__main__":
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/imdb_reviews.csv")
    parser.add_argument("--out_dir", default="artifacts")
    args = parser.parse_args()
    main(args.csv, args.out_dir)
