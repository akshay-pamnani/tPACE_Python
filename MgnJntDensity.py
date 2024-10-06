import numpy as np
import sys
import os
sys.path.append(os.path.abspath('src'))
from trapzRcpp import trapz
from NormKernel import NormKernel

# Uniform distribution support function
def dunif(x, lower, upper):
    return np.where((x >= lower) & (x <= upper), 1/(upper - lower), 0)

def P0(X, supp=None):
    n, d = X.shape
    if supp is None:
        supp = np.tile(np.array([0, 1]), (d, 1))

    tmp = np.ones(n)
    for j in range(d):
        tmp *= dunif(X[:, j], supp[j, 0], supp[j, 1]) * (supp[j, 1] - supp[j, 0])
    
    return np.mean(tmp)

def Pj(j, x, X, h=None, K='epan', supp=None):
    N, d = x.shape
    n = X.shape[0]

    if K != 'epan':
        print('Epanechnikov kernel is only supported currently. It uses Epanechnikov kernel automatically')
        K = 'epan'
    
    if supp is None:
        supp = np.tile(np.array([0, 1]), (d, 1))
    
    if h is None:
        h = np.tile(0.25 * n ** (-1 / 5), d) * (supp[:, 1] - supp[:, 0])

    tmpIndex = np.ones(n)
    for l in range(d):
        tmpIndex *= dunif(X[:, l], supp[l, 0], supp[l, 1]) * (supp[l, 1] - supp[l, 0])

    index = np.where(tmpIndex == 1)[0]
    pHat = np.sum(NormKernel(x[:, j], X[:, j], h[j], K, supp[j])[:, index], axis=1) / n


    pHat /= trapz(np.sort(x[:, j]), pHat[np.argsort(x[:, j])])

    return pHat / P0(X, supp)

def Pkj(kj, x, X, h=None, K='epan', supp=None):
    N, d = x.shape
    n = X.shape[0]

    if K != 'epan':
        print('Epanechnikov kernel is only supported currently. It uses Epanechnikov kernel automatically')
        K = 'epan'

    if supp is None:
        supp = np.tile(np.array([0, 1]), (d, 1))
    
    if h is None:
        h = np.tile(0.25 * n ** (-1 / 5), d) * (supp[:, 1] - supp[:, 0])

    k = kj[0]
    pHatk = NormKernel(x[:, k], X[:, k], h[k], K, supp[k])

    j = kj[1]
    pHatj = NormKernel(x[:, j], X[:, j], h[j], K, supp[j])

    tmpIndex = np.ones(n)
    for l in range(d):
        tmpIndex *= dunif(X[:, l], supp[l, 0], supp[l, 1]) * (supp[l, 1] - supp[l, 0])

    index = np.where(tmpIndex == 1)[0]
    pHat = np.dot(pHatk[:, index], pHatj[:, index].T) / n

    pHat /= trapz(np.sort(x[:, j]), Pj(j, x, X, h, K, supp)[np.argsort(x[:, j])])
    pHat /= trapz(np.sort(x[:, k]), Pj(k, x, X, h, K, supp)[np.argsort(x[:, k])])

    return pHat / P0(X, supp)

def MgnJntDensity(x, X, h=None, K='epan', supp=None):
    N, d = x.shape
    n = X.shape[0]

    if K != 'epan':
        print('Epanechnikov kernel is only supported currently. It uses Epanechnikov kernel automatically')
        K = 'epan'

    if supp is None:
        supp = np.tile(np.array([0, 1]), (d, 1))
    
    if h is None:
        h = np.tile(0.25 * n ** (-1 / 5), d) * (supp[:, 1] - supp[:, 0])

    pMatMgn = np.zeros((N, d))
    pArrJnt = np.zeros((N, N, d, d))

    for j in range(d):
        pMatMgn[:, j] = Pj(j, x, X, h, K, supp)

        for k in range(j, d):
            if k == j:
                pArrJnt[:, :, k, j] = np.diag(pMatMgn[:, j])
            else:
                pArrJnt[:, :, k, j] = Pkj([k, j], x, X, h, K, supp)
                pArrJnt[:, :, j, k] = pArrJnt[:, :, k, j].T

    return {'pArrJnt': pArrJnt, 'pMatMgn': pMatMgn}

