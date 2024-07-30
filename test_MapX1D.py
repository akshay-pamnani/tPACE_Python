import numpy as np
from MapX1D import map_x_1d

def test_basic_arguments():
    xn = np.array([1, 2, 3, 4, 16])
    y = np.array([[i + j*15 for j in range(2)] for i in range(1, 16)])
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16])
    
    expected_matrix = np.array([[1, 16], [2, 17], [3, 18], [4, 19], [15, 30]])
    result_matrix = map_x_1d(x, y, xn)
    np.testing.assert_array_equal(result_matrix, expected_matrix)

    expected_vector = np.linspace(0, 1, 14)[:4]
    result_vector = map_x_1d(np.arange(1, 15), np.linspace(0, 1, 14), np.arange(1, 5))
    np.testing.assert_array_equal(result_vector, expected_vector)
