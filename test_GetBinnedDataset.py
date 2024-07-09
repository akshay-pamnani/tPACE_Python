import pytest
import numpy as np
from GetBinnedDataset import get_binned_dataset
from SetOptions import set_options

def test_trivial_examples():
    y = [np.arange(1, 1001), np.arange(3, 1013)]
    t = [np.linspace(0, 1, 1000), np.linspace(0, 1, 1010)]
    A = get_binned_dataset(y, t, optns=set_options(y, t, None))

    assert len(A['newt'][1]) == 400
    assert pytest.approx(A['newt'][1][112], 0.01) == 0.2807
    assert pytest.approx(np.mean(A['newt'][0])) == np.mean(A['newt'][1])
    assert pytest.approx(A['newy'][0][312], 0.01) == 782
    assert pytest.approx(np.mean(A['newy'][0])) == 500.5

def test_binned_examples_with_singleton():
    y = [1, np.linspace(0, 1000, 1000), np.linspace(3, 1012, 1010)]
    t = [0.5, np.linspace(0, 1, 1000), np.linspace(0, 1, 1010)]
    A = get_binned_dataset(y, t, optns=set_options(y, t, {'numBins': 20}))

    assert len(A['newt'][2]) == 20
    assert abs(A['newt'][1][9] - A['newy'][1][9] / 1000) < 1e-3
    assert pytest.approx(np.mean(A['newt'][2])) == np.mean(A['newt'][1])

if __name__ == '__main__':
    pytest.main()
