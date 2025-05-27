import numpy as np
import pandas as pd

def get_cov_dense(ymat, mu, optns):
    """
    Calculate the sample covariance matrix for dense, regular functional data.

    Parameters:
    - ymat: np.ndarray, shape (n, p) - matrix of dense regular functional data.
    - mu: np.ndarray, shape (p,) - estimated cross-sectional mean vector.
    - optns: dict - options containing:
        * 'dataType' (str): Must be "Dense" or "DenseWithMV".
        * 'userMu' (optional, np.ndarray): If provided, adjusts ymat by subtracting this vector.
        * 'error' (bool): If True, adjusts diagonal for variance.
        * 'userSigma2' (optional, float): User-provided variance for diagonal adjustment.
    
    Returns:
    - dict with keys:
        * 'rawCov': None (since it's not computed in this function)
        * 'smoothCov': np.ndarray - sample covariance matrix on observed grid.
        * 'bwCov': None
        * 'sigma2': float - estimated variance if 'error' is True; None otherwise.
        * 'outGrid': None
    """
    
    if optns['dataType'] not in ['Dense', 'DenseWithMV']:
        raise ValueError("Sample Covariance is only applicable for dataType='Dense' or 'DenseWithMV'.")

    n, m = ymat.shape

    # Adjust ymat by subtracting mu if 'userMu' is provided in options
    if optns.get('userMu') is not None:
        ymat = ymat - np.tile(mu, (n, 1))  # Repeat mu across rows
        K = np.zeros((m, m))
        
        # Compute the covariance matrix manually while handling NaNs
        for i in range(m):
            for j in range(m):
                XcNaNindx = np.isnan(ymat[:, i])
                YcNaNindx = np.isnan(ymat[:, j])
                NaNrows = np.where(XcNaNindx | YcNaNindx)[0]
                indx = np.setdiff1d(np.arange(n), NaNrows)
                K[i, j] = np.sum(ymat[indx, i] * ymat[indx, j]) / (n - 1 - len(NaNrows))
    else:
        # Use pairwise complete observations to calculate covariance if 'userMu' is not provided
        K = np.cov(ymat, rowvar=False, bias=False)
    
    # Ensure symmetry of K
    K = 0.5 * (K + K.T)
    
    # Check for any NaN in the covariance matrix
    if np.isnan(K).any():
        raise ValueError("Data is too sparse to be considered DenseWithMV. Remove sparse observations or specify dataType='Sparse' for FPCA.")
    
    sigma2 = None
    if optns.get('error', False):
        # Use the 2nd order difference method for estimating variance, if not provided
        if 'userSigma2' in optns:
            sigma2 = optns['userSigma2']
        else:
            ord_diff = 2
            sigma2 = np.mean(np.diff(ymat, n=ord_diff, axis=1)**2, where=~np.isnan(ymat)) / np.math.comb(2 * ord_diff, ord_diff)
            np.fill_diagonal(K, np.diag(K) - sigma2)
    
    # Create return dictionary with similar structure to SmoothCov object in R
    ret = {
        'rawCov': None,
        'smoothCov': K,
        'bwCov': None,
        'sigma2': sigma2,
        'outGrid': None
    }
    
    return ret
