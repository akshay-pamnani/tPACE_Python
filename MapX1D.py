import numpy as np
from scipy.interpolate import interp1d

def map_x_1d(x, y, newx):
    if isinstance(y, np.ndarray) and y.ndim == 1:
        # y is a vector
        interpolator = interp1d(x, y, kind='linear', fill_value='extrapolate')
        newy = interpolator(newx)
    elif isinstance(y, np.ndarray) and y.ndim == 2:
        # y is a matrix
        newy = np.apply_along_axis(lambda yy: interp1d(x, yy, kind='linear', fill_value='extrapolate')(newx), axis=0, arr=y)
    else:
        raise ValueError("y must be a vector or a matrix")
    
    if np.isnan(newy).any():
        raise ValueError("NA values found during the mapping from (x, y) to (newx, newy)")

    return newy


