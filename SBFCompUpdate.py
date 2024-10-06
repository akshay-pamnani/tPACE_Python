import numpy as np
from CompFntCent import CompFntCent
from CondProjection import CondProjection
from MgnJntDensity import P0, dunif

def SBFCompUpdate(f, ind, fNW, Y, X, x, h=None, K='epan', supp=None, MgnJntDens=None):
    """
    Smooth backfitting update for a component function.
    
    Parameters:
    f (np.ndarray): Current SBF estimator of component functions (N*d matrix).
    ind (int): Index of updating component during SBF algorithm.
    fNW (np.ndarray): Marginal regression function kernel estimators at estimation points (N*d matrix).
    Y (np.ndarray): Response observation points (n-dim. vector).
    X (np.ndarray): Covariate observation grid (n*d matrix).
    x (np.ndarray): Estimation grid (N*d matrix).
    h (np.ndarray): Bandwidths (d-dim. vector).
    K (str): Kernel function (default 'epan').
    supp (np.ndarray): Supports of estimation of interest (d*2 matrix).
    MgnJntDens (dict): Evaluated values of marginal and 2-dim. joint densities.

    Returns:
    np.ndarray: Updated smooth backfitting component functions for the designated component (N*d matrix).
    """

    N = x.shape[0]
    d = x.shape[1]
    n = X.shape[0]
    
    # Kernel check, currently supporting only Epanechnikov
    if K != 'epan':
        print("Epanechnikov kernel is the default choice")
        K = 'epan'
    
    # Set default support if not provided
    if supp is None:
        supp = np.tile([0, 1], (d, 1))
    
    # Set default bandwidth if not provided
    if h is None:
        h = np.full(d, 0.25 * n**(-1/5)) * (supp[:, 1] - supp[:, 0])
    
    # tmpIndex: Ensure X values are within the support range
    tmpIndex = np.ones(n)
    for l in range(d):
        tmpIndex *= dunif(X[:, l], supp[l, 0], supp[l, 1]) * (supp[l, 1] - supp[l, 0])
    tmpIndex = np.where(tmpIndex == 1)[0]
    
    # Compute the mean of Y for the valid indices
    yMean = np.sum(Y[tmpIndex]) / len(Y) / P0(X, supp)
    
    j = ind
    tmp1, tmp2 = 0, 0
    
    # Update the j-th component function
    if j == 1:
        for k in range(j + 1, d):
            tmp2 += CondProjection(f, [k, j], x, X, MgnJntDens)
        f[:, j] = fNW[:, j] - yMean - tmp2
    elif 1 < j < d:
        for k in range(1, j):
            tmp1 += CondProjection(f, [k, j], x, X, MgnJntDens)
        for k in range(j + 1, d):
            tmp2 += CondProjection(f, [k, j], x, X, MgnJntDens)
        f[:, j] = fNW[:, j] - yMean - tmp1 - tmp2
    elif j == d:
        for k in range(1, d):
            tmp1 += CondProjection(f, [k, j], x, X, MgnJntDens)
        f[:, d] = fNW[:, d] - yMean - tmp1

    # Center the updated component function
    f[:, j] = CompFntCent(f, j, x, MgnJntDens)
    
    return f
