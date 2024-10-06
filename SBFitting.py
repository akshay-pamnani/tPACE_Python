import numpy as np
from MgnJntDensity import MgnJntDensity, dunif
from SBFCompUpdate import SBFCompUpdate
from NWMgnReg import nw_mgn_reg

def SBFitting(Y, x, X, h=None, K='epan', supp=None):
    
    if len(x.shape) == 1:
        print('Evaluation grid must be multi-dimensional.')
        return None
    
    if len(X.shape) == 1:
        print('Observation grid must be multi-dimensional.')
        return None
    
    N = x.shape[0]
    n = X.shape[0]
    d = X.shape[1]
    
    if K != 'epan':
        print('Epanechnikov kernel is only supported currently. It uses Epanechnikov kernel automatically')
        K = 'epan'
    
    if supp is None:
        supp = np.tile([0, 1], (d, 1))
    
    if h is None:
        h = 0.25 * n**(-1/5) * (supp[:, 1] - supp[:, 0])
    
    if len(h) < 2:
        print('Bandwidth must be multi-dimensional.')
        return None
    
    tmpIndex = np.ones(n, dtype=bool)
    for l in range(d):
        tmpIndex = tmpIndex * dunif(X[:, l], supp[l, 0], supp[l, 1]) * (supp[l, 1] - supp[l, 0])

    
    yMean = np.sum(Y[tmpIndex]) / len(Y) / P0(X, supp=supp)
    
    MgnJntDens = MgnJntDensity(x, X, h, K, supp)
    fNW = nw_mgn_reg(Y, x, X, h, K, supp)
    
    f = np.zeros((N, d))
    
    # backfitting
    eps = 100
    iter_count = 1
    critEps = 5e-5
    critEpsDiff = 5e-4
    critIter = 50
    
    while eps > critEps:
        
        f0 = f.copy()
        
        for j in range(d):
            f[:, j] = SBFCompUpdate(f, j, fNW, Y, X, x, h, K, supp, MgnJntDens)[:, j]
            
            nan_indices = np.isnan(f[:, j])
            if np.any(nan_indices):
                f[nan_indices, j] = 0
            
            if np.sum(f[:, j] * f0[:, j]) < 0:
                f[:, j] = -f[:, j]
        
        eps = np.max(np.sqrt(np.mean((f - f0)**2, axis=0)))
        
        if np.abs(eps - eps) < critEpsDiff:
            return {
                'SBFit': f,
                'mY': yMean,
                'NW': fNW,
                'mgnDens': MgnJntDens['pMatMgn'],
                'jntDens': MgnJntDens['pArrJnt'],
                'iterNum': iter_count,
                'iterErr': eps,
                'iterErrDiff': eps - eps,
                'critNum': critIter,
                'critErr': critEps,
                'critErrDiff': critEpsDiff
            }
        
        if iter_count > critIter:
            print('The algorithm may not converge (SBF iteration > stopping criterion). Try another choice of bandwidths.')
            return {
                'SBFit': f,
                'mY': yMean,
                'NW': fNW,
                'mgnDens': MgnJntDens['pMatMgn'],
                'jntDens': MgnJntDens['pArrJnt'],
                'iterNum': iter_count,
                'iterErr': eps,
                'iterErrDiff': np.abs(eps - eps),
                'critNum': critIter,
                'critErr': critEps,
                'critErrDiff': critEpsDiff
            }
        
        iter_count += 1
    
    return {
        'SBFit': f,
        'mY': yMean,
        'NW': fNW,
        'mgnDens': MgnJntDens['pMatMgn'],
        'jntDens': MgnJntDens['pArrJnt'],
        'iterNum': iter_count,
        'iterErr': eps,
        'iterErrDiff': np.abs(eps - eps),
        'critNum': critIter,
        'critErr': critEps,
        'critErrDiff': critEpsDiff
    }
