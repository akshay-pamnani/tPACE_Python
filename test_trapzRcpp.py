import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.abspath('src'))
from trapzRcpp import trapz

def test_trapzRcpp_trivial_example():
    x = np.array([0, 2])
    y = np.array([0, 2])
    assert np.isclose(trapz(x, y), 2)

def test_trapzRcpp_nearly_trivial_example():
    x = np.linspace(0, 4, 100)
    y = x + np.sin(x)
    assert np.isclose(trapz(x, y), 9.653418652171286)
