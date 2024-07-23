import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, squareform

def get_user_cov(optns, obs_grid, cut_reg_grid, buff, ymat):
    """
    Converts the user-provided covariance function into a smoothed covariance matrix.
    
    Parameters:
    - optns: A dictionary containing user covariance function details and other options.
    - obs_grid: Observation grid (not used in this conversion, but kept for completeness).
    - cut_reg_grid: Grid points where the covariance should be evaluated.
    - buff: Buffer to check range coverage.
    - ymat: Matrix for error variance estimation.
    
    Returns:
    - A dictionary containing 'rawCov', 'smoothCov', 'bwCov', 'sigma2', and 'outGrid'.
    """

    # Convert user covariance inputs to numpy arrays
    t = np.array(optns['userCov']['t'], dtype=float)
    cov = np.array(optns['userCov']['cov'], dtype=float)
    
    # Check if user-provided range covers the support of the data
    range_user = (t.min(), t.max())
    range_cut = (cut_reg_grid.min(), cut_reg_grid.max())
    
    if range_user[0] > range_cut[0] + buff or range_user[1] < range_cut[1] - buff:
        raise ValueError('The range defined by the user-provided covariance does not cover the support of the data.')
    
    # Interpolate the user-provided covariance to the cutRegGrid
    interp_function = interp1d(t, cov, kind='linear', fill_value='extrapolate')
    smooth_cov = interp_function(cut_reg_grid)
    
    # Create symmetric smooth covariance matrix
    smooth_cov_matrix = squareform(pdist(cut_reg_grid.reshape(-1, 1), metric='euclidean'))
    smooth_cov_matrix = smooth_cov_matrix @ smooth_cov.reshape(-1, 1)
    smooth_cov_matrix = (smooth_cov_matrix + smooth_cov_matrix.T) / 2
    
    # Calculate sigma2 if error is True
    if optns.get('error', False):
        if 'userSigma2' in optns:
            sigma2 = optns['userSigma2']
        elif optns['dataType'] in ['Dense', 'DenseWithMV']:
            ord = 2
            diffs = np.diff(ymat, n=ord, axis=0)
            sigma2 = np.mean(np.square(diffs), axis=None) / (2 * ord)
        else:
            raise ValueError('Use GetSmoothedCovarSurface instead!')
    else:
        sigma2 = None
    
    # Prepare results
    res = {
        'rawCov': None,
        'smoothCov': smooth_cov_matrix,
        'bwCov': None,
        'sigma2': sigma2,
        'outGrid': cut_reg_grid
    }
    
    return res
