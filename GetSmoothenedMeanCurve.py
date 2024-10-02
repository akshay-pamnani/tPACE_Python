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

    # Print diagnostic info about t
    print("Type of t:", type(t))
    print("Contents of t:", t)

    # Ensure t is a flat list or array of numbers
    if isinstance(t, (list, np.ndarray)):
        t = np.array(t).flatten()  # Convert to a 1D NumPy array
    else:
        raise ValueError("Expected t to be a list or NumPy array.")

    # Ensure xin is a NumPy array of floats
    xin = np.array(t, dtype=float)
    
    # Continue with the rest of your logic
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
            if method_bw_mu in ['GCV', 'GMeanAndGCV']:
                bw_mu = gcv_lwls_1d(y, t, kernel, npoly, nder)
                if len(bw_mu) == 0:
                    raise ValueError('The data is too sparse to estimate a mean function. Get more data!')
                if method_bw_mu == 'GMeanAndGCV':
                    min_bw = np.min(t)  # Adjust based on your needs
                    bw_mu = np.sqrt(min_bw * bw_mu)
            else:
                bw_mu = CVLwls1D(y, t, kernel, npoly, nder, optns)

        sorted_indices = np.argsort(xin)  # Get the sorted indices
        yin = np.array(y)[sorted_indices]  # Sort y according to xin
        xin = np.sort(xin)  # Sort xin for further processing
        win = np.ones_like(xin)

        # Compute mu using lwls_1d
        mu = lwls_1d(bw_mu, kernel, npoly, nder, xin, yin, obs_grid, win)
        mu_dense = lwls_1d(bw_mu, kernel, npoly, nder, xin, yin, reg_grid, win)

    result = {
        'mu': mu,
        'muDense': mu_dense,
        'bw_mu': bw_mu
    }
    return result

# # Example test data
# y = np.random.rand(10)  # 10 random observations
# t = np.linspace(0, 1, 10)  # 10 time points evenly spaced
# obs_grid = np.linspace(0, 1, 50)  # Observation grid
# reg_grid = np.linspace(0, 1, 50)  # Regular grid
# optns = {
#     'userMu': None,  # No user-defined mean function
#     'methodBwMu': 'GCV',  # Example bandwidth method
#     'userBwMu': -1,  # User bandwidth not specified
#     'kernel': 'epanechnikov'  # Example kernel
# }

# result = get_smoothed_mean_curve(y, t, obs_grid, reg_grid, optns)
# print(result)
