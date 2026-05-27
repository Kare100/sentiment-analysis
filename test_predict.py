"""
Basic unit tests for the sentiment predictor.
Run with: pytest tests/
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_mock_model(prediction=1, probabilities=None):
    """Return a scikit-learn-style mock model."""
    mock = MagicMock()
    mock.predict.return_value = np.array([prediction])
    mock.predict_proba.return_value = np.array(
        [probabilities or ([0.1, 0.9] if prediction == 1 else [0.85, 0.15])]
    )
    return mock


def make_mock_vectorizer():
    mock = MagicMock()
    mock.transform.return_value = MagicMock()
    return mock


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPredict:
    def test_positive_prediction(self):
        from src.predict import predict

        result = predict(
            "This film was absolutely brilliant!",
            model=make_mock_model(prediction=1),
            vectorizer=make_mock_vectorizer(),
        )

        assert result["label"] == "Positive"
        assert result["score"] == 1
        assert 0 < result["confidence"] <= 1.0
        assert "text" in result

    def test_negative_prediction(self):
        from src.predict import predict

        result = predict(
            "Terrible movie, complete waste of time.",
            model=make_mock_model(prediction=0),
            vectorizer=make_mock_vectorizer(),
        )

        assert result["label"] == "Negative"
        assert result["score"] == 0

    def test_confidence_is_float(self):
        from src.predict import predict

        result = predict(
            "It was okay.",
            model=make_mock_model(prediction=1, probabilities=[0.45, 0.55]),
            vectorizer=make_mock_vectorizer(),
        )
        assert isinstance(result["confidence"], float)

    def test_batch_returns_list(self):
        from src.predict import predict_batch

        texts = ["Great!", "Awful.", "Not bad."]
        results = predict_batch(
            texts,
            model=make_mock_model(prediction=1),
            vectorizer=make_mock_vectorizer(),
        )

        assert isinstance(results, list)
        assert len(results) == 3

    def test_result_has_required_keys(self):
        from src.predict import predict

        result = predict(
            "Some text.",
            model=make_mock_model(),
            vectorizer=make_mock_vectorizer(),
        )
        for key in ("text", "label", "confidence", "score"):
            assert key in result, f"Missing key: {key}"
