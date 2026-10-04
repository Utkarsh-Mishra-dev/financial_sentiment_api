"""FastAPI app serving the TF-IDF + Logistic Regression financial
sentiment baseline.

Run locally:
    uvicorn main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

LABELS = ["negative", "neutral", "positive"]
MODEL_DIR = Path(__file__).resolve().parent  # look for .pkl files next to this script

app = FastAPI(
    title="Financial Sentiment Classifier API",
    description="TF-IDF + Logistic Regression model for 3-class financial sentiment "
                 "classification (negative / neutral / positive).",
    version="1.0.0",
)

# Load once at startup, not per-request - loading from disk on every
# call would be slow and wasteful.
vectorizer = None
clf = None
model_loaded = False

try:
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    clf = joblib.load(MODEL_DIR / "logistic_regression.pkl")
    model_loaded = True
except Exception as e:
    # Don't crash the app if the model fails to load - /health will report
    # it instead, which is more useful for debugging a deployed service.
    print(f"Failed to load model: {e}")


class PredictRequest(BaseModel):
    sentence: str = Field(
        ...,
        min_length=1,
        description="A financial news sentence to classify.",
        examples=["The company reported a 20% increase in quarterly profit."],
    )


class PredictResponse(BaseModel):
    sentence: str
    predicted_label: str
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {
        "status": "ok" if model_loaded else "error",
        "model_loaded": model_loaded,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    X = vectorizer.transform([request.sentence])
    pred_idx = clf.predict(X)[0]
    probs = clf.predict_proba(X)[0]

    return PredictResponse(
        sentence=request.sentence,
        predicted_label=LABELS[pred_idx],
        probabilities={LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
    )
