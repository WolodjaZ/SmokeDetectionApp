import numpy as np

from src import utils


def test_save_and_load_dict():
    pass


def test_set_seed():
    utils.set_seeds()
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    utils.set_seeds()
    x = np.random.randn(2, 3)
    y = np.random.randn(2, 3)
    assert np.array_equal(a, x)
    assert np.array_equal(b, y)
