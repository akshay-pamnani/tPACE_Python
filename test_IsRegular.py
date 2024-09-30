import pytest
from isRegular import isRegular

def test_isregular_dense():
    assert isRegular([list(range(1, 11)), list(range(1, 11)), list(range(1, 11))]) == 'Dense'

def test_isregular_dense_with_mv():
    assert isRegular([list(range(1, 4)), list(range(1, 5)), list(range(1, 5))]) == 'DenseWithMV'

def test_isregular_sparse():
    assert isRegular([list(range(1, 3)), list(range(1, 4)), list(range(1, 5))]) == 'Sparse'

def test_isregular_dense_but_irregular():
    assert isRegular([list(range(1, 4)) + [5], list(range(1, 4)) + [5], list(range(1, 4)) + [5]]) == 'Sparse'
