import numpy as np

from src import evaluate


def test_get_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    performance = evaluate.get_metrics(y_true=y_true, y_pred=y_pred)
    assert performance["balanced_accuracy"] == 1 / 2
    assert performance["f1"] == 1 / 2
    assert performance["precision"] == 1 / 2
    assert performance["recall"] == 1 / 2
