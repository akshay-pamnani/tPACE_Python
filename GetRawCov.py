import numpy as np
from scipy import linalg

def uniqueM(x):
    """ Helper function to map values in x to unique integers """
    unique_vals = np.unique(x)
    id1 = np.zeros(len(x), dtype=int)
    for i, val in enumerate(unique_vals):
        id1[np.where(x == val)[0]] = i + 1
    return id1

def meshgrid(x, y):
    """ Custom meshgrid function to replicate R's meshgrid functionality """
    X, Y = np.meshgrid(x, y)
    return {'X': X, 'Y': Y}

def GetRawCov(y, t, obsGridnew, mu, dataType, error):
    """
    Obtain raw covariance.
    
    Parameters:
    - y: list of n arrays (repeated measurements for n subjects)
    - t: list of n arrays (time points for n subjects)
    - obsGridnew: array of m time points corresponding to mu
    - mu: array of fitted mean functions (corresponding to pooled unique time points from t)
    - dataType: output of IsRegular() (should be one of 'Sparse', 'DenseWithMV', 'Dense', 'RegularWithMV')
    - error: boolean flag (True if measurement error assumption is applied, False otherwise)

    Returns:
    A dictionary containing:
    - tPairs: (N, 2) matrix of pairs of time points for subjects
    - cxxn: 1D array of raw covariance corresponding to tPairs
    - indx: 1D array of indices for each subject
    - win: 1D array of weights for 2-D smoother (if required)
    - cyy: 1D array of raw covariance for all pairs of time points
    - diag: 2-column matrix for raw covariance along diagonal if error == True
    """
    
    ncohort = len(y)
    obsGrid = np.sort(np.unique(np.concatenate(t)))  # sort and flatten the time points
    mu_interpolated = np.interp(obsGrid, obsGridnew, mu)  # interpolate mu to match obsGrid
    count = None
    indx = None
    diag = None

    if dataType in ['Sparse', 'DenseWithMV']:
        Ys = [meshgrid(yi, t[i]) for i, yi in enumerate(y)]
        Xs = [meshgrid(ti, t[i]) for i, ti in enumerate(t)]

        # Vectorize the grids for y & t
        xx1 = np.concatenate([x['X'].flatten() for x in Xs])
        xx2 = np.concatenate([x['Y'].flatten() for x in Xs])
        yy2 = np.concatenate([y['Y'].flatten() for y in Ys])
        yy1 = np.concatenate([y['X'].flatten() for y in Ys])

        # Get id1/2 such that xx1/2 = q(id1/2), where q = unique(xx1/2)
        id1 = uniqueM(xx1)
        id2 = uniqueM(xx2)
        cyy = (yy1 - mu_interpolated[id1]) * (yy2 - mu_interpolated[id2])

        # Index for subject i
        indx = np.repeat(np.arange(len(y)), [len(yi) ** 2 for yi in y])

        tPairs = np.column_stack([xx1, xx2])

        if error:
            tneq = np.where(xx1 != xx2)[0]
            teq = np.where(xx1 == xx2)[0]
            indx = indx[tneq]
            diag = np.column_stack([tPairs[teq, 0], cyy[teq]])
            tPairs = tPairs[tneq]
            cxxn = cyy[tneq]
        else:
            cxxn = cyy

    elif dataType == 'Dense':
        yy = np.array([np.ravel(yi) for yi in y]).T
        MU = np.tile(mu, (len(y), 1)).T
        t1 = t[0]

        yy = yy - MU
        cyy = np.dot(yy.T, yy) / ncohort
        cyy = cyy.flatten()

        cxxn = cyy
        xxyy = meshgrid(t1, t1)  # Create meshgrid for t1

        tPairs = np.column_stack([xxyy['X'].flatten(), xxyy['Y'].flatten()])

        if error:
            tneq = np.where(tPairs[:, 0] != tPairs[:, 1])[0]
            teq = np.where(tPairs[:, 0] == tPairs[:, 1])[0]
            diag = np.column_stack([tPairs[teq, 0], cyy[teq]])
            tPairs = tPairs[tneq]
            cxxn = cyy[tneq]
        else:
            cxxn = cyy

    elif dataType == 'RegularWithMV':
        raise ValueError("This is not implemented yet. Contact Pantelis!")

    else:
        raise ValueError("Invalid 'dataType' argument type")

    result = {
        'tPairs': tPairs,
        'cxxn': cxxn,
        'indx': indx,
        'cyy': cyy,
        'diag': diag,
        'count': count,
        'error': error,
        'dataType': dataType
    }

    return result
