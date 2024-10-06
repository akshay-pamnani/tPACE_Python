import numpy as np
import sys
import os
sys.path.append(os.path.abspath('src'))
from trapzRcpp import trapz

def CondProjection(f, kj, x, X, MgnJntDens):
    """
    Conditional projection of the k-th component function on the j-th component function space.

    Parameters:
    f (np.ndarray): Evaluated values of component functions at the estimation grid (N*d matrix).
    kj (list): Index of conditional projection for the k-th component function on the j-th component function space (2-dim. vector).
    x (np.ndarray): Estimation grid (N*d matrix).
    X (np.ndarray): Covariate observation grid (n*d matrix).
    MgnJntDens (dict): Evaluated values of marginal and 2-dim. joint densities (2-dim. list).

    Returns:
    np.ndarray or int: Conditional projection of the k-th component function on the j-th component function space (N-dim. vector).
    """

    N = x.shape[0]
    n = X.shape[0]
    d = X.shape[1]
    
    k = kj[0]
    j = kj[1]

    xj = x[:, j]
    
    # Determine whether to use X or x for the k-th component
    if len(f[:, k]) == n:
        xk = X[:, k]
    else:
        xk = x[:, k]

    asdf = MgnJntDens['pMatMgn'][:, j]
    
    tmpInd = np.where(asdf != 0)[0]
    qwer = MgnJntDens['pArrJnt'][:, tmpInd, k, j]

    if len(tmpInd) > 0:
        pHat = np.zeros((len(xk), len(xj)))
        pHat[:, tmpInd] = qwer / asdf[tmpInd]

        tmp = []
        for l in range(pHat.shape[1]):
            tmptmp = f[:, k] * pHat[:, l]
            tmp.append(trapz(tmptmp[np.argsort(xk)], np.sort(xk)))

        return np.array(tmp)
    else:
        return 0
