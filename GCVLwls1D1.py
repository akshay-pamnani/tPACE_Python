import numpy as np
from Minb import Minb
from Lwls1D import lwls_1d


def gcv_lwls_1d(yy, tt, kernel, npoly, nder, data_type, verbose=True):
    # Check if yy and tt are vectors and "cheat" if necessary
    if isinstance(yy, np.ndarray) and isinstance(tt, np.ndarray) and not isinstance(yy, list) and not isinstance(tt, list):
        if len(tt) < 21:
            raise ValueError("You are trying to use a local linear weight smoother in a vector with less than 21 values.")
        
        my_partition = np.concatenate((np.arange(1, 11), np.random.choice(10, len(tt) - 10, replace=True)))
        yy = np.array_split(yy, my_partition)
        tt = np.array_split(tt, my_partition)
        data_type = 'Sparse'

    t = np.concatenate(tt)
    y = np.concatenate(yy)[np.argsort(t)]

    t = np.sort(t)
    N = len(t)
    r = t[-1] - t[0]

    # Starting bandwidth candidates
    if data_type == "Sparse":
        dstar = Minb(t, npoly + 2)
        if dstar > r * 0.25:
            dstar = dstar * 0.75
            print(f"Warning: The min bandwidth choice is too big, reduced to {dstar}!")

        h0 = 2.5 * dstar
    elif data_type == "DenseWithMV":
        h0 = 2.0 * Minb(t, npoly + 1)
    else:
        h0 = 1.5 * Minb(t, npoly + 1)

    if np.isnan(h0):
        if kernel == "gauss":
            h0 = 0.2 * r
        else:
            raise ValueError("The data is too sparse, no suitable bandwidth can be found! Try Gaussian kernel instead!")

    h0 = min(h0, r)
    q = (r / (4 * h0)) ** (1 / 9)
    bw_candidates = np.sort(q ** np.arange(10) * h0)

    # Unique indices
    idx = np.unique(t, return_index=True)[1]

    k0_candidates = {
        'quar': 0.9375, 'epan': 0.7500, 'rect': 0.5000,
        'gausvar': 0.498677, 'gausvar1': 0.598413, 'gausvar2': 0.298415, 'other': 0.398942
    }

    k0 = k0_candidates.get(kernel, k0_candidates['other'])
    gcv_scores = []

    # Compute GCV scores
    for bw in bw_candidates:
        newmu = lwls_1d(bw, kernel, npoly=npoly, nder=nder, xin=t, yin=y, win=np.ones(len(y)), xout=np.sort(np.unique(t)))
        cvsum = np.sum((newmu[idx] - y) ** 2)
        gcv_score = cvsum / (1 - (r * k0) / (N * bw)) ** 2
        gcv_scores.append(gcv_score)

    gcv_scores = np.array(gcv_scores)

    # If no finite gcvScore, increase bandwidth and retry
    if np.all(np.isinf(gcv_scores)):
        bw_candidates = np.linspace(max(bw_candidates), r, num=2 * len(bw_candidates))
        for bw in bw_candidates:
            newmu = lwls_1d(bw, kernel, npoly=npoly, nder=nder, xin=t, yin=y, win=np.ones(len(y)), xout=np.sort(np.unique(t)))
            cvsum = np.sum((newmu[idx] - y) ** 2)
            gcv_score = cvsum / (1 - (r * k0) / (N * bw)) ** 2
            gcv_scores.append(gcv_score)

    gcv_scores = np.array(gcv_scores)

    # If the problem persists, data is too sparse
    if np.all(np.isinf(gcv_scores)):
        raise ValueError("The data is too sparse, no suitable bandwidth can be found! Try Gaussian kernel instead!")

    bInd = np.argmin(gcv_scores)
    bScr = gcv_scores[bInd]
    bOpt = bw_candidates[bInd]

    if bOpt == r:
        print("Warning: Data is too sparse, optimal bandwidth includes all the data! You may want to change to Gaussian kernel!")

    return {'bOpt': bOpt, 'bScore': bScr}

# Note: Implement the `min_bandwidth` function based on your specific logic.
