# 🧠 Sentiment Analysis — NLP Text Classifier

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black?logo=flask)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A text sentiment classifier built with **TF-IDF** and **Logistic Regression**, served via a **Flask REST API**. Given any English text, it predicts whether the sentiment is **positive** or **negative** along with a confidence score.

---

## ✨ Features

- **TF-IDF vectorization** with bigram support
- **Logistic Regression** classifier (fast, interpretable, strong baseline)
- **REST API** via Flask with single and batch prediction endpoints
- **Confusion matrix** visualization saved automatically after training
- Clean project structure with unit tests

---

## 📁 Project Structure

```
sentiment-analysis/
├── src/
│   ├── train.py        # Train and save the model
│   └── predict.py      # Load model and run inference
├── tests/
│   └── test_predict.py # Unit tests (pytest)
├── notebooks/          # Exploratory analysis (Jupyter)
├── models/             # Saved model artifacts (auto-created)
├── data/               # Place your dataset here
├── app.py              # Flask REST API
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone & install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train.py
```

This saves `models/sentiment_model.pkl` and `models/tfidf_vectorizer.pkl`.

### 3. Run the API

```bash
python app.py
```

API is live at `http://localhost:5000`.

---

## 📡 API Usage

### Single prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely incredible!"}'
```

**Response:**
```json
{
  "text": "This movie was absolutely incredible!",
  "label": "Positive",
  "confidence": 0.9341,
  "score": 1
}
```

### Batch prediction

```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great film!", "Total waste of time."]}'
```

### Health check

```bash
curl http://localhost:5000/health
```

---

## 🧪 Run Tests

```bash
pytest tests/ -v
```

---

## 🔧 How It Works

1. **Vectorization** — Raw text is converted to a TF-IDF matrix (top 5,000 features, unigrams + bigrams, stopwords removed).
2. **Classification** — A Logistic Regression model is trained on the vectorized text.
3. **Inference** — New text is transformed using the same vectorizer and passed to the model, which returns a class label and probability.

---

## 📈 Extending This Project

- Swap in a **transformer model** (e.g. `distilbert-base-uncased`) via HuggingFace 🤗
- Add a **Streamlit** front-end for a live demo
- Train on the full **IMDB 50K dataset** from Kaggle
- Deploy to **Render**, **Railway**, or **Hugging Face Spaces**

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
