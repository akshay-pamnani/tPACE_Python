import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def create_folds(y, k=10):
    n = len(y)
    if n == 0:
        raise ValueError('Response length is zero')

    uniq_y = np.unique(y)
    if not pd.api.types.is_categorical_dtype(y) and len(y) / len(uniq_y) >= k:
        # Intepret the integer-valued y as class labels. Stratify if the number of class labels is <= 5.
        y = pd.Categorical(y)
    elif pd.api.types.is_numeric_dtype(y):
        # 5-stratum Stratified sampling
        if n >= 5 * k:
            breaks = np.unique(np.quantile(y, np.linspace(0, 1, num=5)))
            y = pd.cut(y, bins=breaks, include_lowest=True, labels=False)
        else:
            y = np.ones(len(y), dtype=int)

    samp_list = {label: simple_folds(np.where(y == label)[0], k) for label in np.unique(y)}
    list0 = [[] for _ in range(k)]
    samp = [list1 + list2 for list1, list2 in zip(list0, map(list, zip(*samp_list.values())))]

    return samp

def simple_folds(yy, k=10):
    if len(yy) > 1:
        all_samp = shuffle(yy)
    else:
        all_samp = yy

    n = len(yy)
    n_each = n // k
    samp = [all_samp[i * n_each:(i + 1) * n_each] for i in range(k)]

    rest_samp = all_samp[n_each * k:]
    rest_ind = shuffle(np.arange(k))[:len(rest_samp)]

    for i, ind in enumerate(rest_ind):
        samp[ind] = np.concatenate([samp[ind], [rest_samp[i]]])

    return samp

