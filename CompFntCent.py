import numpy as np
import sys
import os
sys.path.append(os.path.abspath('src'))
from trapzRcpp import trapz

def CompFntCent(f, j, x, MgnJntDens):
    """
    Centering component functions using the marginal mean.

    Parameters:
    f (np.ndarray): Evaluated values of component functions at the estimation grid (N*d matrix).
    j (int): Index of centering for the j-th component function.
    x (np.ndarray): Estimation grid (N*d matrix).
    MgnJntDens (dict): Evaluated values of marginal and 2-dim. joint densities (dictionary with 'pMatMgn').

    Returns:
    np.ndarray: Centered component function values at the estimation grid (N-dim. vector).
    """
    
    fj = f[:, j]
    xj = x[:, j]

    # Extract the marginal density values for the j-th component
    pMatMgn = MgnJntDens['pMatMgn'][:, j]
    
    # Compute the centered component
    tmp = fj - trapz(fj * pMatMgn[np.argsort(xj)], np.sort(xj))
    
    return tmp
