import numpy as np
from scipy.interpolate import interp1d, interp2d,griddata

def convert_support(from_grid, to_grid, mu=None, Cov=None, phi=None, is_cross_cov=False):
    if mu is not None:
        return map_x1d(from_grid, mu, to_grid)
    elif Cov is not None:
        # Assuming Cov is a numpy array or matrix
        f = interp2d(from_grid, from_grid, Cov, kind='linear')
        return f(to_grid, to_grid)
    elif phi is not None:
        return map_x1d(from_grid, phi, to_grid)
    else:
        raise ValueError("One of mu, Cov, or phi must be provided.")
    
def map_x1d(from_grid, data, to_grid):
    # Interpolate 1D data from from_grid to to_grid
    f = interp1d(from_grid, data, kind='linear', fill_value='extrapolate')
    return f(to_grid)

def map_x1d_matrix(from_grid, matrix, to_grid):
    # Interpolate each column of the matrix from from_grid to to_grid
    interpolated_matrix = np.zeros((len(to_grid), matrix.shape[1]))
    for i in range(matrix.shape[1]):
        f = interp1d(from_grid, matrix[:, i], kind='linear', fill_value='extrapolate')
        interpolated_matrix[:, i] = f(to_grid)
    return interpolated_matrix

def interpolate_covariance(from_grid, Cov, to_grid):
    # Interpolate covariance matrix Cov from from_grid to to_grid
    points = np.array(np.meshgrid(from_grid, from_grid)).T.reshape(-1, 2)
    values = Cov.flatten()
    grid_x, grid_y = np.meshgrid(to_grid, to_grid)
    interpolated_cov = griddata(points, values, (grid_x, grid_y), method='linear')
    return interpolated_cov
