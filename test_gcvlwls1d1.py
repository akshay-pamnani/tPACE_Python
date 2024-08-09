import pytest
import numpy as np
import rdata
from GCVLwls1D1 import gcv_lwls_1d


@pytest.fixture
def data():
    converted = rdata.read_rda("dataForGcvLwls.RData")
    t = converted['t']
    y = converted['y']
    return y,t


def test_basic_optimal_bandwidth_choice_for_sparse_data_epan(data):
    y, t = data
    A = gcv_lwls_1d(y, t, 'epan', 1, 0, 'Sparse')
    assert np.isclose(A['bOpt'], 2.071354057811459, atol=1e-7)

def test_basic_optimal_bandwidth_choice_for_sparse_data_rect(data):
    y, t = data
    B = gcv_lwls_1d(y, t, 'rect', 1, 0, 'Sparse')
    assert np.isclose(B['bOpt'], 2.238990337557121, atol=0.04)


