"""
Sentiment Analysis Model Training
Uses TF-IDF + Logistic Regression for text classification.
Dataset: IMDB movie reviews (via sklearn or manual CSV)
"""

import os
import pickle
import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_sample_data():
    """Load a small built-in dataset for demo purposes (no download needed)."""
    texts = [
        "This movie was absolutely amazing, loved every minute of it!",
        "Terrible film, waste of time and money.",
        "Great acting and storyline, highly recommend.",
        "Boring and predictable, nothing new here.",
        "One of the best movies I have ever seen!",
        "Awful plot, bad acting, completely disappointed.",
        "A masterpiece! The director did a wonderful job.",
        "Not worth watching, very disappointing experience.",
        "Incredible visuals and a gripping story from start to finish.",
        "Slow pacing and poor character development ruined it for me.",
        "A fantastic film with outstanding performances by the cast.",
        "Dreadful movie, would not recommend to anyone.",
        "Heartwarming and beautifully crafted, a true gem.",
        "The worst movie I have seen this year, total disappointment.",
        "Stunning cinematography and an emotional rollercoaster!",
        "Overhyped and underwhelming, did not meet expectations at all.",
        "Absolutely loved it! Will watch again for sure.",
        "Dull, lifeless, and forgettable. Avoid this one.",
        "A refreshing take on the genre, full of surprises.",
        "Poorly written script with zero character depth.",
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    return texts, labels


def train_model(texts, labels):
    """Train a TF-IDF + Logistic Regression sentiment classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n✅ Model trained successfully!")
    print(f"   Accuracy: {acc:.2%}")
    print(f"\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    return model, vectorizer, X_test_vec, y_test, y_pred


def save_model(model, vectorizer, output_dir="models"):
    """Persist model and vectorizer to disk."""
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/sentiment_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{output_dir}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"\n💾 Model saved to '{output_dir}/'")


def plot_confusion_matrix(y_test, y_pred, output_dir="models"):
    """Save confusion matrix as PNG."""
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    print(f"📈 Confusion matrix saved to '{output_dir}/confusion_matrix.png'")


if __name__ == "__main__":
    print("🚀 Loading data...")
    texts, labels = load_sample_data()

    print("🔧 Training model...")
    model, vectorizer, X_test_vec, y_test, y_pred = train_model(texts, labels)

    save_model(model, vectorizer)
    plot_confusion_matrix(y_test, y_pred)
