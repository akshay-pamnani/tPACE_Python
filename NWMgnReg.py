import numpy as np
from NormKernel import NormKernel
from MgnJntDensity import dunif

def nw_mgn_reg(Y, x, X, h=None, K='epan', supp=None):
    """
    Nadaraya-Watson marginal regression estimation.

    Parameters:
    Y : np.ndarray
        Response observation points (n-dim. vector).
    x : np.ndarray
        Estimation grid (N*d matrix).
    X : np.ndarray
        Covariate observation grid (n*d matrix).
    h : np.ndarray, optional
        Bandwidths (d-dim. vector).
    K : str, optional
        Kernel function (default is 'epan').
    supp : np.ndarray, optional
        Supports of estimation interested (d*2 matrix).

    Returns:
    np.ndarray
        NW marginal regression function kernel estimators at each estimation point (N*d matrix).
    """
    N, d = x.shape
    n, _ = X.shape

    # Default kernel is Epanechnikov
    if K != 'epan':
        print('Epanechnikov kernel is only supported. Using Epanechnikov kernel.')
        K = 'epan'
    
    # Set default support if not provided
    if supp is None:
        supp = np.tile([0, 1], (d, 1))
    
    # Set default bandwidths if not provided
    if h is None:
        h = 0.25 * n**(-1/5) * (supp[:, 1] - supp[:, 0])

    fNW = np.zeros((N, d))  # Initialize output matrix

    # Calculate weights based on the uniform distribution and supports
    tmpIndex = np.ones(n)
    for j in range(d):
        tmpIndex *= dunif(X[:, j], supp[j, 0], supp[j, 1]) * (supp[j, 1] - supp[j, 0])
    tmpIndex = np.where(tmpIndex == 1)[0]

    # Nadaraya-Watson estimation for each component function
    for j in range(d):

        pHatj = NormKernel(x[:, j], X[:, j], h[j], K, (supp[j, 0], supp[j, 1]))

        # print(f"Debug: pHatj shape for component {j}: {pHatj.shape}")
        
        # print(f"Debug: tmpIndex for component {j}: {tmpIndex}")
        
        rHatj = (pHatj[:, tmpIndex] @ Y[tmpIndex]) / len(Y)
        
        # print(f"Debug: rHatj for component {j}: {rHatj.shape}, Value: {rHatj}")

        pHatj = np.sum(pHatj[:, tmpIndex], axis=1) / len(Y)
        
        # print(f"Debug: pHatj_sum for component {j}: {pHatj.shape}, Value: {pHatj}")

        tmpInd = np.where(pHatj != 0)[0]
        
        # print(f"Debug: Non-zero indices for pHatj_sum for component {j}: {tmpInd}")

        # If tmpInd is not empty, show its values
        # if len(tmpInd) > 0:
        #     print(f"Debug: Values for valid indices in rHatj: {rHatj[tmpInd]}, pHatj_sum: {pHatj[tmpInd]}")
        # else:
        #     print(f"Debug: No valid indices for component {j} to update fNW.")
        
        fNW[tmpInd, j] = rHatj[tmpInd] / pHatj[tmpInd]

    return fNW
