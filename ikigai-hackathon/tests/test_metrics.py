"""
Tests for metrics.
"""

import pytest
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def test_metrics_on_perfect_prediction():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    assert accuracy_score(y_true, y_pred) == 1.0
    assert precision_score(y_true, y_pred) == 1.0
    assert recall_score(y_true, y_pred) == 1.0
    assert f1_score(y_true, y_pred) == 1.0
    assert roc_auc_score(y_true, y_score) == 1.0


def test_metrics_on_wrong_prediction():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])

    assert accuracy_score(y_true, y_pred) == 0.0
    assert roc_auc_score(y_true, y_score) == 0.0


def test_metrics_binary():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.4, 0.6, 0.8])

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred)

    assert 0 <= acc <= 1
    assert 0 <= prec <= 1
    assert 0 <= rec <= 1
    assert 0 <= f1 <= 1
    assert 0 <= auc <= 1
    assert cm.shape == (2, 2)


def test_confusion_matrix_values():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    cm = confusion_matrix(y_true, y_pred)
    assert cm[0, 0] == 1  # TN
    assert cm[0, 1] == 1  # FP
    assert cm[1, 0] == 1  # FN
    assert cm[1, 1] == 1  # TP


if __name__ == "__main__":
    pytest.main([__file__, "-v"])