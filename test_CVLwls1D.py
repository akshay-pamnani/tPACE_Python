import pytest
import numpy as np
import rdata
from CVLwls1D import CVLwls1D


@pytest.fixture
def data():
    converted = rdata.read_rda("dataGeneratedByExampleSeed123.RData")
    t = converted['t']
    y = converted['y']
    return y,t


def test_CVLwls1D(data):
    y, t = data
    a_result = CVLwls1D(y, t=t, kernel='epan', npoly=1, nder=0, dataType='Sparse')
    expected_result = 4.172873877723954
    tolerance = 0.6
    assert abs(a_result - expected_result) <= tolerance

