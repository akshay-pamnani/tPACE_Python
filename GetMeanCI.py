import numpy as np
import pandas as pd
from List2Mat import list_to_mat
from GetMeanDense import get_mean_dense
from GetSmoothenedMeanCurve import get_smoothed_mean_curve
from SetOptions import set_options


def get_mean_ci(Ly, Lt, level=0.95, R=999, optns=None):
    if optns is None:
        optns = {}
    optns = set_options(Ly, Lt, optns)
    
    # Ensure level is a single value between 0 and 1
    if isinstance(level, (list, tuple)) and len(level) > 1:
        level = level[0]
        print("Warning: The input level has more than 1 element; only the first one is used.")
    if not (0 <= level <= 1):
        raise ValueError("Invalid input value of level.")
    if not isinstance(R, int) or R <= 0:
        raise ValueError("R should be a positive integer.")
    
    n = len(Ly)
    
    if optns['methodMuCovEst'] == 'smooth':
        obs_grid = np.sort(np.unique(np.concatenate(Lt)))
        reg_grid = np.linspace(np.min(obs_grid), np.max(obs_grid), optns['nRegGrid'])
        
        mu_mat = np.array([
            # Bootstrap sampling
            get_smoothed_mean_curve([Ly[i] for i in np.random.choice(n, n, replace=True)],
                                    [Lt[i] for i in np.random.choice(n, n, replace=True)],
                                    obs_grid, 
                                    reg_grid[np.logical_and(reg_grid >= np.min(obs_grid), 
                                                            reg_grid <= np.max(obs_grid))], 
                                    optns)['muDense']
            for _ in range(R)
        ])
        
        # Get the grid indices that don't contain NaN values
        valid_grid_indices = ~np.isnan(mu_mat).any(axis=0)
        CI = np.percentile(mu_mat[:, valid_grid_indices], [100 * (1 - level) / 2, 100 * (1 - (1 - level) / 2)], axis=0)
        CI_grid = reg_grid[valid_grid_indices]
    
    elif optns['methodMuCovEst'] == 'cross-sectional':
        ymat = list_to_mat(Ly, Lt)
        obs_grid = np.sort(np.unique(np.concatenate(Lt)))
        
        mu_mat = np.array([
            # Bootstrap sampling for cross-sectional mean
            get_mean_dense(ymat[np.random.choice(n, n, replace=True), :], obs_grid, optns)['mu']
            for _ in range(R)
        ])
        
        CI = np.percentile(mu_mat, [100 * (1 - level) / 2, 100 * (1 - (1 - level) / 2)], axis=0)
        CI_grid = obs_grid
    
    if optns['dataType'] == 'Sparse':
        print("Warning: Bootstrap CIs for the mean function may not be computed for the entire time range.")
    
    # Construct the data frame with confidence intervals
    CI_df = pd.DataFrame({
        'CIgrid': CI_grid,
        'lower': CI[0, :],
        'upper': CI[1, :]
    })
    
    return {'CI': CI_df, 'level': level}
