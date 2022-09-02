import numpy as np
import pytest

from src import predict


@pytest.mark.parametrize(
    "threshold, y_pred",
    [
        (0.5, [1, 0]),
        (0.6, [0, 0]),
        (0.75, [0, 0]),
    ],
)
def test_custom_predict(threshold, y_pred):
    y_prob = np.array([0.6, 0.4])
    assert np.array_equal(
        predict.custom_predict(y_prob=y_prob, threshold=threshold), np.array(y_pred)
    )
