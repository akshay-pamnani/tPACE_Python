import numpy as np
from scipy.stats import uniform

def norm_kernel(x, X, h, K, supp):
    """Calculate the normal kernel values for the given inputs."""
    # Currently only Epanechnikov kernel is supported
    if K != 'epan':
        print('Epanechnikov kernel is only supported. Using Epanechnikov kernel.')
        K = 'epan'
    
    # Ensure x and X are numpy arrays
    x = np.asarray(x)
    X = np.asarray(X)

    # Compute the Epanechnikov kernel
    u = (x[:, None] - X) / h  # Broadcasting to compute the difference
    kernel_values = 0.75 * (1 - u ** 2) * (np.abs(u) <= 1)  # Epanechnikov kernel formula
    return kernel_values

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
        tmpIndex *= uniform.pdf(X[:, j], loc=supp[j, 0], scale=supp[j, 1] - supp[j, 0]) * (supp[j, 1] - supp[j, 0])
    tmpIndex = np.where(tmpIndex == 1)[0]

    # Nadaraya-Watson estimation for each component function
    for j in range(d):
        pHatj = norm_kernel(x[:, j], X[:, j], h[j], K, (supp[j, 0], supp[j, 1]))
        rHatj = np.sum(pHatj[:, tmpIndex] @ Y[tmpIndex]) / len(Y)
        
        pHatj = np.sum(pHatj[:, tmpIndex], axis=1) / len(Y)
        
        tmpInd = np.where(pHatj != 0)[0]
        
        fNW[tmpInd, j] = rHatj[tmpInd] / pHatj[tmpInd]

    return fNW
