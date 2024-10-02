import numpy as np
from scipy.interpolate import interp1d
from Lwls1D import lwls_1d
from GCVLwls1D1 import gcv_lwls_1d
from CVLwls1D import CVLwls1D

def get_smoothed_mean_curve(y, t, obs_grid, reg_grid, optns):
    user_mu = optns['userMu']
    method_bw_mu = optns['methodBwMu']
    npoly = 1
    nder = 0
    user_bw_mu = optns['userBwMu']
    kernel = optns['kernel']

    # Ensure t is a flat list or array of numbers
    if isinstance(t, (list, np.ndarray)):
        if isinstance(t[0], (list, np.ndarray)):
            t = np.concatenate([np.array(i).flatten() for i in t])
        else:
            t = np.array(t).flatten()
    else:
        raise ValueError("Expected t to be a list or NumPy array.")

    xin = np.array(t, dtype=float)

    # Flatten the y array
    yin = np.concatenate([np.array(arr).flatten() for arr in y])

    # Check for empty xin or yin
    if len(xin) == 0 or len(yin) == 0:
        raise ValueError("xin or yin is empty. Ensure valid data is provided.")

    # Ensure xin is one-dimensional
    if xin.ndim == 0:
        raise ValueError("Expected xin to be a one-dimensional array.")

    if isinstance(user_mu, dict) and 'mu' in user_mu and 't' in user_mu:
        buff = np.finfo(float).eps * max(np.abs(obs_grid)) * 10
        range_user = (min(user_mu['t']), max(user_mu['t']))
        range_obs = (min(obs_grid), max(obs_grid))
        if range_user[0] > range_obs[0] + buff or range_user[1] < range_obs[1] - buff:
            raise ValueError('The range defined by the user provided mean does not cover the support of the data.')

        mu_interp = interp1d(user_mu['t'], user_mu['mu'], kind='linear', fill_value="extrapolate")
        mu = mu_interp(obs_grid)
        mu_dense_interp = interp1d(obs_grid, mu, kind='linear', fill_value="extrapolate")
        mu_dense = mu_dense_interp(reg_grid)
        bw_mu = None
    else:
        if user_bw_mu > 0:
            bw_mu = user_bw_mu
        else:
            # Check if there are enough data points
            if len(t) < 21:
                raise ValueError('Not enough data points. At least 21 are required for GCV.')

            if method_bw_mu in ['GCV', 'GMeanAndGCV']:
                bw_mu = gcv_lwls_1d(yin, t, kernel, npoly, nder, data_type='Sparse')
                if len(bw_mu) == 0:
                    raise ValueError('The data is too sparse to estimate a mean function. Get more data!')
                if method_bw_mu == 'GMeanAndGCV':
                    min_bw = np.min(t)  # Adjust based on your needs
                    bw_mu = np.sqrt(min_bw * bw_mu)
            else:
                bw_mu = CVLwls1D(yin, t, kernel, npoly, nder, optns)

        # Sort xin and yin using the sorted indices
        sorted_indices = np.argsort(xin)
        xin = xin[sorted_indices]  # Sort xin
        yin = yin[sorted_indices]  # Sort yin

        mu = lwls_1d(bw_mu, kernel, npoly, nder, xin, yin, obs_grid, np.ones_like(xin))
        mu_dense = lwls_1d(bw_mu, kernel, npoly, nder, xin, yin, reg_grid, np.ones_like(xin))

    result = {
        'mu': mu,
        'muDense': mu_dense,
        'bw_mu': bw_mu
    }
    return result

# Example test data
# y = np.random.rand(50)  # 50 random observations
# t = np.linspace(0, 1, 50)  # 50 time points evenly spaced
# obs_grid = np.linspace(0, 1, 100)  # Observation grid with 100 points for smoother output
# reg_grid = np.linspace(0, 1, 100)  # Regular grid with 100 points for smoother output
# optns = {
#     'userMu': None,  # No user-defined mean function
#     'methodBwMu': 'GCV',  # Example bandwidth method
#     'userBwMu': -1,  # User bandwidth not specified
#     'kernel': 'epanechnikov'  # Example kernel
# }

# try:
#     result = get_smoothed_mean_curve(y, t, obs_grid, reg_grid, optns)
#     print(result)
# except ValueError as e:
#     print("Error:", e)
