import warnings
import numpy as np

def Minb(x, num_points, legacy_code=False):
    """
    Find the minimum bandwidth choice where the local window contains at least `num_points` points.

    Parameters:
    x (list or numpy array): n x 1 vector
    num_points (int): Number of points in a local window
                      For local weighted constant, num_points is at least 1
                      For local weighted linear, num_points is at least 2
    legacy_code (bool): Flag to use legacy code logic

    Returns:
    float: The minimum bandwidth choice for vector x
    """
    n = len(x)
    
    if num_points < 1 or num_points > n:
        warnings.warn("Invalid number of minimum points specified")
        return float('nan')
    
    if legacy_code:
        x = np.sort(np.unique(x))  # Ensure unique and sorted values
        if num_points > 1:
            return max(x[num_points-1:] - x[:n-num_points+1])
        else:
            return max((x[1:] - x[:-1]) / 2)
    
    grid_pts = np.sort(np.unique(x))
    l,r = 0,num_points
    dist_nn1 = 0
    while r < len(grid_pts):
        dist_nn1 = max(dist_nn1,grid_pts[r] - grid_pts[l])
        r += 1
        l += 1

    
    return dist_nn1


