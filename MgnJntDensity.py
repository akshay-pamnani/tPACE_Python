import numpy as np
from scipy.integrate import trapz

# Uniform distribution support function
def dunif(x, lower, upper):
    return np.where((x >= lower) & (x <= upper), 1/(upper - lower), 0)

# Epanechnikov kernel
def epanechnikov_kernel(x, h, supp):
    lower, upper = supp
    return np.where((x >= lower) & (x <= upper), (3 / 4) * (1 - ((x - h) ** 2)), 0)

# Proportion of non-truncated observation
def P0(X, supp=None):
    n, d = X.shape
    
    if supp is None:
        supp = np.tile([0, 1], (d, 1))
    
    tmp = np.ones(n)
    for j in range(d):
        tmp *= dunif(X[:, j], supp[j, 0], supp[j, 1]) * (supp[j, 1] - supp[j, 0])
    
    return np.mean(tmp)

# Marginal density estimation
def Pj(j, x, X, h=None, K='epan', supp=None):
    N, d = x.shape
    n = X.shape[0]
    
    if K != 'epan':
        print("Epanechnikov kernel is only supported. Using Epanechnikov kernel.")
        K = 'epan'
        
    if supp is None:
        supp = np.tile([0, 1], (d, 1))
    
    if h is None:
        h = np.repeat(0.25 * n ** (-1/5), d) * (supp[:, 1] - supp[:, 0])

    tmpIndex = np.ones(n)
    for l in range(d):
        tmpIndex *= dunif(X[:, l], supp[l, 0], supp[l, 1]) * (supp[l, 1] - supp[l, 0])
        
    index = np.where(tmpIndex == 1)[0]
    pHat = np.sum(epanechnikov_kernel(x[:, j], X[:, j], h[j])[:, index], axis=1) / n
    pHat /= trapz(np.sort(x[:, j]), pHat[np.argsort(x[:, j])])
    
    return pHat / P0(X, supp)

# 2-dimensional joint density estimation
def Pkj(kj, x, X, h=None, K='epan', supp=None):
    N, d = x.shape
    n = X.shape[0]
    
    if K != 'epan':
        print("Epanechnikov kernel is only supported. Using Epanechnikov kernel.")
        K = 'epan'
        
    if supp is None:
        supp = np.tile([0, 1], (d, 1))
    
    if h is None:
        h = np.repeat(0.25 * n ** (-1/5), d) * (supp[:, 1] - supp[:, 0])
    
    k = kj[0]
    pHatk = epanechnikov_kernel(x[:, k], X[:, k], h[k])
    
    j = kj[1]
    pHatj = epanechnikov_kernel(x[:, j], X[:, j], h[j])
    
    tmpIndex = np.ones(n)
    for l in range(d):
        tmpIndex *= dunif(X[:, l], supp[l, 0], supp[l, 1]) * (supp[l, 1] - supp[l, 0])
        
    index = np.where(tmpIndex == 1)[0]
    pHat = np.dot(pHatk[:, index], pHatj[:, index].T) / n
    pHat /= trapz(np.sort(x[:, j]), Pj(j, x, X, h, K, supp)[np.argsort(x[:, j])])
    pHat /= trapz(np.sort(x[:, k]), Pj(k, x, X, h, K, supp)[np.argsort(x[:, k])])
    
    return pHat / P0(X, supp)

# Construction of evaluation matrices for marginal and joint densities estimators
def MgnJntDensity(x, X, h=None, K='epan', supp=None):
    N, d = x.shape
    n = X.shape[0]
    
    if K != 'epan':
        print("Epanechnikov kernel is only supported. Using Epanechnikov kernel.")
        K = 'epan'
        
    if supp is None:
        supp = np.tile([0, 1], (d, 1))
    
    if h is None:
        h = np.repeat(0.25 * n ** (-1/5), d) * (supp[:, 1] - supp[:, 0])

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
