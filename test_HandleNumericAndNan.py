import pytest
import numpy as np
from HandleNumericsAndNan import handle_numerics_and_nan



def test_handle_numerics_and_nan_no_nan_values():
    # Test that NaN values are properly handled
    Ly_nan = [np.array([1, 2, np.nan, 4]), np.array([5, np.nan, 7, 8])]
    Lt_nan = [np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.5, 0.6, np.nan, 0.8])]
    result_nan = handle_numerics_and_nan(Ly_nan, Lt_nan)
    assert np.isnan(result_nan['Ly'][0]).sum() == 0  # No NaN values in Ly
    assert np.isnan(result_nan['Lt'][1]).sum() == 0  # No NaN values in Lt

def test_handle_numerics_and_nan_subjects_with_nan():
    # Test that ValueError is raised for subjects with only NaN values
    Ly_all_nan = [np.array([np.nan, np.nan, np.nan]), np.array([1, 2, 3])]
    Lt_all_nan = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
    with pytest.raises(ValueError):
        handle_numerics_and_nan(Ly_all_nan, Lt_all_nan)


def test_handle_numerics_and_nan_output_values():
    # Test that the output contains numeric arrays with correct values
    Ly_values = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    Lt_values = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
    result_values = handle_numerics_and_nan(Ly_values, Lt_values)
    assert all(isinstance(arr, np.ndarray) for arr in result_values['Ly'])  # Ly contains numpy arrays
    assert all(isinstance(arr, np.ndarray) for arr in result_values['Lt'])  # Lt contains numpy arrays
    assert np.array_equal(result_values['Ly'][0], Ly_values[0])  # Ly contains correct values
    assert np.array_equal(result_values['Lt'][1], Lt_values[1])  # Lt contains correct values

if __name__ == "__main__":
    test_handle_numerics_and_nan_output_values()
    print("All tests passed!")
