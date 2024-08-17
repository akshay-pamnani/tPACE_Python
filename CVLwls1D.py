import numpy as np
import pandas as pd
from Minb import Minb
from CreateFolds import create_folds
from Lwls1D import lwls_1d


def CVLwls1D(y, t, kernel, npoly, nder, dataType, kFolds=5, useBW1SE=False):
    # If 'y' and 't' are vectors, "cheat" and break them into a list of 10 elements
    if isinstance(y, np.ndarray) and isinstance(t, np.ndarray):
        if len(t) < 21:
            raise ValueError("You are trying to use a local linear weight smoother in a vector with less than 21 values.")
        partition = np.concatenate([np.arange(10), np.random.choice(10, len(t) - 10)])
        y = np.array_split(y, partition)
        t = np.array_split(t, partition)
        dataType = 'Sparse'
    
    # Make everything into vectors
    ncohort = len(t)
    tt = np.concatenate(t)
    yy = np.concatenate(y)
    ind = np.concatenate([np.full(len(ti), i) for i, ti in enumerate(t)])
    yyn = yy[np.argsort(tt)]
    ind = ind[np.argsort(tt)]
    ttn = np.sort(tt)

    # Get minimum reasonable bandwidth
    a0 = ttn[0]
    b0 = ttn[-1]
    rang = b0 - a0
    dstar = Minb(tt, npoly + 2)
    h0 = 2.5 * dstar if dataType != 'Dense' else dstar
    if h0 > rang / 4:
        h0 *= 0.75
        print(f"Warning: the min bandwidth choice is too big, reduce to {h0:.2f}!")

    # Get the candidate bandwidths
    nbw = 11
    bw = np.zeros(nbw - 1)
    n = len(np.unique(tt))
    for i in range(nbw - 1):
        bw[i] = 2.5 * rang / n * (n / 2.5) ** ((i) / (nbw - 1))
    bw = bw - np.min(bw) + h0

    # Cross-validation
    cv = np.zeros((kFolds, len(bw)))
    theFolds = create_folds(ind, k=kFolds)

    for j in range(nbw - 1):
        for i in range(kFolds):
            test_indices = theFolds[i]
            xout = ttn[np.isin(ind, test_indices)]
            obs = yyn[np.isin(ind, test_indices)]
            xin = ttn[~np.isin(ind, test_indices)]
            yin = yyn[~np.isin(ind, test_indices)]

            win = np.ones(len(yin))

            try:
                mu = lwls_1d(bw=bw[j], kernel_type=kernel, npoly=npoly, nder=nder, xin=xin, yin=yin, xout=xout, win=win)
            except Exception as e:
                print(f"Warning: Invalid bandwidth during CV. Try enlarging the window size. Error: {e}")
                mu = np.inf

            cv[i, j] = np.sum((obs - mu) ** 2)
            if np.isnan(cv[i, j]):
                cv[i, j] = np.inf

    if np.min(cv) == np.inf:
        raise ValueError("All bandwidths resulted in infinite CV costs.")

    if useBW1SE:
        mean_cv = np.mean(cv, axis=0)
        std_cv = np.std(cv, axis=0)
        bopt = bw[np.max(np.where(mean_cv < np.min(mean_cv) + std_cv[np.argmin(mean_cv)] / np.sqrt(kFolds)))]
    else:
        bopt = bw[np.argmin(np.mean(cv, axis=0))]

    return bopt


