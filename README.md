# Sentiment Analysis

A simple NLP project that classifies text as positive or negative. Built using TF-IDF for feature extraction and Logistic Regression for classification, with a small Flask API to serve predictions. This is one of my first ML projects so feedback is welcome.

## Project structure

```
sentiment-analysis/
├── src/
│   ├── train.py        # trains and saves the model
│   └── predict.py      # loads the model and runs inference
├── tests/
│   └── test_predict.py
├── app.py              # Flask API
├── requirements.txt
└── README.md
```

## Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/Kare100/sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt
```

## Usage

Train the model first:

```bash
python src/train.py
```

Then start the API:

```bash
python app.py
```

The API runs at `http://localhost:5000`.

To get a prediction:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely incredible!"}'
```

Response:

```json
{
  "text": "This movie was absolutely incredible!",
  "label": "Positive",
  "confidence": 0.9341,
  "score": 1
}
```

You can also send a batch of texts to `/predict/batch` or hit `/health` to check the API is running.

## How it works

Text goes through a TF-IDF vectorizer (top 5000 features, unigrams and bigrams, stopwords removed) and then into a Logistic Regression classifier. After training, both the model and vectorizer are saved as `.pkl` files so they can be loaded without retraining.

## Running tests

```bash
pytest tests/ -v
```

## What I want to add next

- train on the full IMDB 50K dataset from Kaggle
- try a transformer model like distilbert
- add a simple Streamlit front-end
