import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.abspath('src'))
from RcppPseudoApprox import PseudoApprox

def test_RcppPseudoApprox_trivial_example():
    np.random.seed(111)
    z = np.random.uniform(0, 1, 44)
    
    # Testing the interpolation function
    result = PseudoApprox(np.array([0, 1]), np.array([0, 2]), z)
    
    # Expected result is 2 * z
    expected = 2 * z
    np.testing.assert_allclose(result, expected, rtol=1e-7)

def test_RcppPseudoApprox_wrong_data():
    with pytest.raises(RuntimeError, match="Unequal vector sizes for linear interpolation"):
        PseudoApprox(np.array([0, 1]), np.array([0, 2, 4]), np.array([0]))

if __name__ == "__main__":
    pytest.main()
