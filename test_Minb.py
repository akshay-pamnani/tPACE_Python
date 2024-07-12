import pytest
import numpy as np
from Minb import Minb

def test_min_b():
    print("\nTests for 'min_bandwidth'")

    assert np.isnan(Minb([1, 2, 3, 4.1], -1))
    assert Minb([1, 2, 3, 4.1], 1) == pytest.approx(2 * 0.55)
    assert np.isnan(Minb([11, 2, 3, 4.1], -1))
    assert Minb([11, 2, 3, 4.1], 2) == pytest.approx(8)
    assert np.isnan(Minb([1, 2, 3, 4.1], 6))



