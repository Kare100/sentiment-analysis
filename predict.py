"""
Sentiment Predictor — load trained model and run inference.
"""

import pickle


def load_artifacts(model_dir="models"):
    with open(f"{model_dir}/sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{model_dir}/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict(text: str, model=None, vectorizer=None) -> dict:
    """
    Predict sentiment of a single text string.

    Returns:
        dict with keys: label (str), confidence (float), score (int)
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_artifacts()

    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]

    label = "Positive" if prediction == 1 else "Negative"
    confidence = float(max(probabilities))

    return {
        "text": text,
        "label": label,
        "confidence": round(confidence, 4),
        "score": int(prediction),
    }


def predict_batch(texts: list, model=None, vectorizer=None) -> list:
    """Run prediction on a list of texts."""
    if model is None or vectorizer is None:
        model, vectorizer = load_artifacts()
    return [predict(t, model, vectorizer) for t in texts]


if __name__ == "__main__":
    samples = [
        "This is hands down the best film I have seen this year!",
        "Boring, predictable, and a total waste of two hours.",
        "An okay movie — not great, not terrible.",
    ]

    print("🔍 Running predictions...\n")
    for result in predict_batch(samples):
        emoji = "😊" if result["label"] == "Positive" else "😞"
        print(f"{emoji} [{result['label']} — {result['confidence']:.0%}] \"{result['text']}\"")
