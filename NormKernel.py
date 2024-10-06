import numpy as np
import sys
import os
sys.path.append(os.path.abspath('src'))
from trapzRcpp import trapz
from ScaleKernel import ScaleKernel, dunif

def NormKernel(x, X, h=None, K='epan', supp=None):
    N = len(x)
    n = len(X)

    if K != 'epan':
        print('Epanechnikov kernel is the default choice.')
        K = 'epan'
    
    if supp is None:
        supp = np.array([0, 1])
    
    if h is None:
        h = 0.25 * n**(-1/5) * (supp[1] - supp[0])

    numer = ScaleKernel(x, X, h, K=K, supp=supp)

    # Setting elements where X is outside the support to zero
    ind1 = np.where(dunif(X, supp[0], supp[1]) == 0)[0]
    numer[:, ind1] = 0

    denom = np.zeros(n)
    for i in range(n):
        denom[i] = trapz(np.sort(x), numer[np.argsort(x), i])

    ind2 = np.where(denom == 0)[0]

    NormKernelTmp = numer / denom[np.newaxis, :]  # Broadcasting the denom
    NormKernelTmp[:, ind2] = 0

    if min(NormKernelTmp.shape) == 1:
        return NormKernelTmp.flatten()  # Return as a 1D array if it's one-dimensional
    else:
        return NormKernelTmp
