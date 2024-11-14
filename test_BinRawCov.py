##### The Testing values need to be revisited #####

import numpy as np
import pytest
from BinRawCov import BinRawCov

def test_bin_raw_cov_no_error():
    # Define `tPairs` and example `rcov` dictionary
    tPairs = np.array([[1, 1], [2, 1], [2, 1], [2, 2], [1, 2], [1, 2], [1, 2]])
    rcov = {
        "tPairs": tPairs,
        "cxxn": np.arange(1, tPairs.shape[0] + 1),  # Equivalent to 1:nrow(tPairs) in R
        "error": False
    }
    
    # Call the `BinRawCov` function
    brcov = BinRawCov(rcov)
    
    # Expected results
    expected_tPairs = np.array([[1, 1],  [1, 2], [2, 1],[2, 2]]) # np.array([[1, 1],  [2, 1], [1, 2],[2, 2]])
    expected_meanVals = np.array([1, 6, 2.5, 4]) # np.array([1, 2.5, 6, 4])
    
    # Assertions
    np.testing.assert_array_equal(brcov.tPairs, expected_tPairs)
    np.testing.assert_array_almost_equal(brcov.meanVals, expected_meanVals)

def test_bin_raw_cov_with_error():
    # Define `tPairs` and example `rcov` dictionary
    tPairs = np.array([[1, 1], [2, 1], [2, 1], [2, 2], [1, 2], [1, 2], [1, 2]])
    rcov = {
        "tPairs": tPairs,
        "cxxn": np.arange(1, tPairs.shape[0] + 1),  # Equivalent to 1:nrow(tPairs) in R
        "error": True
    }
    
    # Call the `BinRawCov` function
    brcov = BinRawCov(rcov)
    
    # Expected results
    expected_tPairs = np.array([[1, 1],  [1, 2], [2, 1],[2, 2]]) # np.array([[1, 1],  [2, 1], [1, 2],[2, 2]])
    expected_meanVals = np.array([1, 6, 2.5, 4]) # np.array([1, 2.5, 6, 4])
    
    # Assertions
    np.testing.assert_array_equal(brcov.tPairs, expected_tPairs)
    np.testing.assert_array_almost_equal(brcov.meanVals, expected_meanVals)

if __name__ == "__main__":
    pytest.main()

