import numpy as np

def list_to_mat(y, t):
    n = len(y)
    obs_grid = sorted(set().union(*t))
    ymat = np.full((n, len(obs_grid)), np.nan)
    
    for i in range(n):
        indices = [obs_grid.index(time_point) for time_point in t[i]]
        ymat[i, indices] = y[i]
    
    return ymat
