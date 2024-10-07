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

    # Sort xj and obtain the sorted indices
    # sorted_indices = np.argsort(xj)
    # xj_sorted = xj[sorted_indices].flatten()
    # fj_sorted = fj[sorted_indices].flatten()
    # pMatMgn_sorted = pMatMgn[sorted_indices].flatten()

    # Create a structured array for sorting
    data = np.column_stack((xj, fj, pMatMgn))

    # Sort the structured array by the first column (xj)
    data_sorted = data[np.argsort(data[:, 0])]

    # Unpack the sorted data
    xj_sorted, fj_sorted, pMatMgn_sorted = data_sorted[:, 0], data_sorted[:, 1], data_sorted[:, 2]


    # Debugging prints
    print("xj_sorted:", xj_sorted)
    print("fj_sorted:", fj_sorted)
    print("pMatMgn_sorted:", pMatMgn_sorted)

    # Check if xj_sorted is sorted
    if not np.all(np.diff(xj_sorted) >= 0):
        print("Warning: xj_sorted is not sorted correctly.")

    # Compute the centered component
    tmp = fj - trapz(xj_sorted, fj_sorted * pMatMgn_sorted)

    return tmp

