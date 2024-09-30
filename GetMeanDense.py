import numpy as np
from scipy.interpolate import interp1d

class SMC:
    """
    A class to represent the SMC object containing the cross-sectional mean function.
    """
    def __init__(self, mu, mu_dense=None, mu_bw=None):
        self.mu = mu
        self.mu_dense = mu_dense
        self.mu_bw = mu_bw

def get_mean_dense(ymat, obs_grid, optns):
    """
    Obtains the cross-sectional mean function at observed grid for dense regular functional data.
    
    Parameters:
    ymat (np.ndarray): Matrix of dense regular functional data.
    obs_grid (np.ndarray): Observed grid for interpolation.
    optns (dict): Options for FPCA function.
    
    Returns:
    SMC: An SMC object containing:
        - mu: p-dim vector of mean function estimation, i.e., on observed grid.
        - mu_dense: None.
        - mu_bw: None.
    """
    # Check optns
    if optns['dataType'] not in ['Dense', 'DenseWithMV']:
        raise ValueError('Cross-sectional mean is only applicable for option: dataType = "Dense" or "DenseWithMV"!')
    
    if 'userMu' not in optns or optns['userMu'] is None:
        mu = np.nanmean(ymat, axis=0)  # Use non-missing data only
    else:
        t = optns['userMu']['t']
        mu_values = optns['userMu']['mu']
        spline = interp1d(t, mu_values, kind='linear', fill_value='extrapolate')
        mu = spline(obs_grid)
    
    if np.any(np.isnan(mu)):
        raise ValueError('The cross-sectional mean appears to have NaN! Consider setting your dataType to \'Sparse\' manually')
    
    return {
        'mu': mu,
        'mu_dense': None,
        'mu_bw': None
    }
    #return SMC(mu=mu)

# Example usage
# ymat = np.array(...) # Your dense regular functional data
# obs_grid = np.array(...) # Your observed grid
# optns = {
#     'dataType': 'Dense',
#     'userMu': {
#         't': np.array(...), 
#         'mu': np.array(...)
#     }
# }
# result = get_mean_dense(ymat, obs_grid, optns)

