"""Trains the TF-IDF + Logistic Regression baseline and exports it as .pkl
files for the FastAPI app to load.

Run locally, from a location where data/processed/train.csv exists
(the same file produced by the main FinBERT project's Step 2):

    python train_baseline.py

Produces:
    tfidf_vectorizer.pkl
    logistic_regression.pkl
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

TRAIN_CSV = Path("train.csv")  # adjust path if needed
LABELS = ["negative", "neutral", "positive"]  # index 0,1,2


def main():
    train_df = pd.read_csv(TRAIN_CSV)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
    X_train = vectorizer.fit_transform(train_df["sentence"])
    y_train = train_df["label"]

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    joblib.dump(clf, "logistic_regression.pkl")
    print("Saved tfidf_vectorizer.pkl and logistic_regression.pkl")


if __name__ == "__main__":
    main()
