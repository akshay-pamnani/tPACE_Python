import numpy as np
import pytest
from scipy.interpolate import interp1d

# Assuming the GetRawCov function has been defined earlier as discussed
# Replace with your actual GetRawCov function here

def test_get_raw_cov_sparse_true():
    # Loading data manually as an example
    # Replace with actual data loading
    y = [...]  # Replace with actual data
    t = [...]  # Replace with actual data
    mu = [...]  # Replace with actual data

    # Equivalent to GetRawCov(y, t, sort(unlist(t)), mu, 'Sparse', TRUE)
    AA = GetRawCov(y, t, np.sort(np.unique(np.concatenate(t))), mu, 'Sparse', True)

    # Test cases for Sparse, True
    assert np.isclose(np.sum(AA['indx']), 184, atol=2 * np.finfo(float).eps)
    assert np.isclose(np.sum(AA['cxxn']), -7.416002855888680, atol=1e-13)
    assert np.isclose(np.sum(AA['cyy']), 16.327874649330514, atol=1e-13)
    assert np.isclose(np.sum(AA['tPairs']), 4.053285461728229e+02, atol=1e-12)

def test_get_raw_cov_sparse_false():
    # Equivalent to GetRawCov(y, t, sort(unlist(t)), mu, 'Sparse', FALSE)
    BB = GetRawCov(y, t, np.sort(np.unique(np.concatenate(t))), mu, 'Sparse', False)

    # Test cases for Sparse, False
    assert np.isclose(np.sum(BB['indx']), 298, atol=2 * np.finfo(float).eps)
    assert np.isclose(np.sum(BB['cxxn']), 16.327874649330514, atol=1e-13)
    assert np.isclose(np.sum(BB['cyy']), 16.327874649330514, atol=1e-13)
    assert np.isclose(np.sum(BB['tPairs']), 6.330209554605514e+02, atol=1e-12)

def test_get_raw_cov_dense_true():
    # Example for Dense data
    y2 = [list(range(1, 11)), list(range(2, 12))]
    t2 = [list(range(1, 11)), list(range(1, 11))]
    mu = np.linspace(1.5, 10.5, 10)  # Sample mu
    CC = GetRawCov(y2, t2, np.sort(np.unique(np.concatenate(t2))), mu, 'Dense', True)

    # Test cases for Dense, True
    assert np.isclose(np.sum(CC['indx']), 0, atol=2 * np.finfo(float).eps)
    assert np.isclose(np.sum(CC['cxxn']), 22.5, atol=1e-13)
    assert np.isclose(np.sum(CC['cyy']), 25, atol=1e-13)
    assert np.isclose(np.sum(CC['tPairs']), 990, atol=1e-12)

def test_get_raw_cov_dense_false():
    # Example for Dense data, False
    DD = GetRawCov(y2, t2, np.sort(np.unique(np.concatenate(t2))), np.linspace(1.5, 10.5, 10), 'Dense', False)

    # Test cases for Dense, False
    assert np.isclose(np.sum(DD['indx']), 0, atol=2 * np.finfo(float).eps)
    assert np.isclose(np.sum(DD['cxxn']), 25, atol=1e-13)
    assert np.isclose(np.sum(DD['cyy']), 25, atol=1e-13)
    assert np.isclose(np.sum(DD['tPairs']), 1100, atol=1e-12)

# To run the tests
if __name__ == "__main__":
    pytest.main()
