"""
Flask REST API for the Sentiment Analysis model.

Endpoints:
  POST /predict       — single text prediction
  POST /predict/batch — batch prediction
  GET  /health        — health check
"""

from flask import Flask, request, jsonify
from src.predict import load_artifacts, predict, predict_batch

app = Flask(__name__)

# Load model once at startup
model, vectorizer = load_artifacts()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "TF-IDF + Logistic Regression"})


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Request body must include a 'text' field."}), 400

    result = predict(data["text"], model, vectorizer)
    return jsonify(result)


@app.route("/predict/batch", methods=["POST"])
def predict_batch_endpoint():
    data = request.get_json()
    if not data or "texts" not in data or not isinstance(data["texts"], list):
        return jsonify({"error": "Request body must include a 'texts' list."}), 400

    results = predict_batch(data["texts"], model, vectorizer)
    return jsonify({"results": results, "count": len(results)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
