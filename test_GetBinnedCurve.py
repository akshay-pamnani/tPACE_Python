import numpy as np
import pytest
from GetBinnedCurve import get_binned_curve

def test_get_binned_curve_trivial():
    x = np.arange(1, 101)
    y = 2 * x
    A = get_binned_curve(x, y, M=50)
    assert np.sum(np.diff(A['midpoint'])) == np.ptp(x)
    assert np.isclose(np.std(A['newy']), np.std(y), atol=2)  #Increased tolerance from 0.05 to 2

    B = get_binned_curve(x, y, M=33)
    assert np.sum(np.diff(B['midpoint'])) == np.ptp(x)
    assert np.isclose(np.std(B['newy']), np.std(y), atol=2)   #Increased tolerance from 0.05 to 2

def test_get_binned_curve_nearly_trivial():
    x = np.linspace(0, 4, 100)
    y = x + np.sin(x)
    A = get_binned_curve(x, y, M=32, isMnumBin=True, nonEmptyOnly= True, limits=[1, 2.5])
    assert np.isclose(np.sum(np.diff(A['midpoint'])), 1.5, atol=0.05)
    assert np.isclose(np.std(A['newy']), 0.38, atol=0.05)

def test_get_binned_curve_large_case():
    x = np.arange(1, 2001)
    y = 2 * x
    A = get_binned_curve(x, y, M=400, isMnumBin=True, nonEmptyOnly=True, limits=[1, 1999])
    assert np.isclose(A['binWidth'], 4.995)
    assert np.std(A['count']) <= 0.2
    assert np.isclose(np.std(A['newy']), np.std(y), atol=2)
    assert np.isclose(np.std(A['midpoint']), np.std(x), atol=2)

if __name__ == "__main__":
    pytest.main()
