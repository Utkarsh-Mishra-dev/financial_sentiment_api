# Financial Sentiment Classifier API

A FastAPI service serving a TF-IDF + Logistic Regression model that classifies
a financial news sentence as **negative**, **neutral**, or **positive**.

This model is the baseline from a larger project
([Financial Sentiment Classification using FinBERT](#)), trained on the
[Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)
dataset (`sentences_50agree` config).

## What the model predicts

Given a single financial news sentence, the model predicts its sentiment
**from an investor's perspective** — i.e. whether the news seems favorable
or unfavorable to the company's stock/shareholders — as one of three
classes: `negative`, `neutral`, `positive`.

## Endpoints

### `GET /health`

Returns service status and whether the model loaded successfully.

**Example response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`

Accepts a sentence and returns the predicted sentiment with class
probabilities.

**Example request body:**
```json
{
  "sentence": "The company reported a 20% increase in quarterly profit."
}
```

**Example response:**
```json
{
  "sentence": "The company reported a 20% increase in quarterly profit.",
  "predicted_label": "positive",
  "probabilities": {
    "negative": 0.0123,
    "neutral": 0.0456,
    "positive": 0.9421
  }
}
```

Interactive API docs (Swagger UI) are available at `/docs` once the service
is running.

## How to run locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train and export the model** (run once, requires
   `data/processed/train.csv` from the main FinBERT project's data
   preparation step):
   ```bash
   python train_baseline.py
   ```
   This produces `tfidf_vectorizer.pkl` and `logistic_regression.pkl` in
   the current directory.

3. **Start the API server:**
   ```bash
   uvicorn main:app --reload
   ```

4. **Test it:**
   ```bash
   curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"sentence": "The company reported a 20% increase in quarterly profit."}'
   ```

   Or visit `http://127.0.0.1:8000/docs` for an interactive interface.

## Deployment

Deployed live on Render: `<add your live URL here after deploying>`

## Model

- **Vectorizer:** `TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=20000)`
- **Classifier:** `LogisticRegression(class_weight="balanced")`
- **Test set performance:** 78.3% accuracy, 0.754 macro F1
  (see the main FinBERT project repo for full evaluation and comparison
  against a fine-tuned Transformer model).
