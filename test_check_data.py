import pytest
import numpy as np
from CheckData import CheckData


def test_check_data_type():
    # Test that y and t are lists
    with pytest.raises(ValueError):
        CheckData([1, 2, 3], 4)

    with pytest.raises(ValueError):
        CheckData(1, [1, 2, 3])

def test_check_data_length():
    # Test that y and t have the same length
    with pytest.raises(ValueError):
        CheckData([1, 2, 3], [1, 2, 3, 4])

'''def test_check_data():
    #Test that all vectors are of the same length
    with pytest.raises(ValueError):
        CheckData([[1, 2], [4, 5, 6]], [[1, 2, 3], [4, 5]])'''


def test_check_data_type_members():
    # Test that y and t members are numeric
    with pytest.raises(ValueError):
        CheckData([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 'a']])

def test_check_data_time_order():
    # Test that t is in ascending order
    with pytest.raises(ValueError):
        CheckData([[1, 2, 3], [4, 3, 2]], [[1, 2, 3], [4, 3, 2]])

def test_check_data_time_duplicates():
    # Test that t does not contain duplicates
    with pytest.raises(ValueError):
        CheckData([[1, 2, 3], [4, 5, 6]], [[1, 1, 2], [4, 5, 6]])

def test_check_data_inf():
    # Test that y does not contain Inf
    with pytest.raises(ValueError):
        CheckData([[1, 2, 3], [4, 5, np.inf]], [[1, 2, 3], [4, 5]])

def test_check_data_time_gap():
    # Test that there is no large time gap
    CheckData([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]])


def test_check_data_no_warnings():
    # Test that there are no warnings for a valid input
    CheckData([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]])



if __name__ == "__main__":
    test_check_time_gaps()
    print("All tests passed!")

    