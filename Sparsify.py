import numpy as np

def sparsify_sample(samp, pts, sparsity, aggressive=False, fragment=False):
    """
    Sparsify densely observed functional data for experimental purposes.
    
    Parameters:
    - samp: np.ndarray - matrix of densely observed functional data; each row is a sample.
    - pts: np.ndarray - vector of grid points corresponding to the columns of samp.
    - sparsity: list or int - list of integers for possible number of observations per sample.
    - aggressive: bool - if True, ensure nearby readings are excluded.
    - fragment: bool or float - if True, fragment the observations. If a float, specifies the approximate length of each fragment.
    
    Returns:
    - dict - a dictionary with keys 'Lt' (list of observation time points for each sample)
             and 'Ly' (list of values corresponding to those time points for each sample).
    """
    
    # Validate input
    if not isinstance(samp, np.ndarray) or samp.ndim != 2:
        raise ValueError('samp needs to be a 2D numpy array (matrix).')
    if samp.shape[1] != len(pts):
        raise ValueError('The number of columns in samp needs to be equal to the length of pts.')
    if isinstance(sparsity, int):
        sparsity = [sparsity, sparsity]

    if aggressive and fragment:
        raise ValueError('Specify only one of `aggressive` or `fragment`.')

    n_samples, n_pts = samp.shape

    if aggressive:
        ind_each = [remote_sampling(n_pts, sparsity) for _ in range(n_samples)]
    elif fragment:
        avg_npts = fragment / np.mean(np.diff(pts))
        ind_each = []
        for _ in range(n_samples):
            ran_pts = (min(pts), max(pts))
            mid = np.random.uniform(ran_pts[0], ran_pts[1])
            use_pts = np.where((pts >= mid - 0.5 * (ran_pts[1] - ran_pts[0]) * fragment) &
                               (pts <= mid + 0.5 * (ran_pts[1] - ran_pts[0]) * fragment))[0]
            n_samp_pts = np.random.choice(sparsity)
            ind_each.append(sorted(np.random.choice(use_pts, min(n_samp_pts, len(use_pts)), replace=False)))
    else:
        ind_each = [sorted(np.random.choice(n_pts, np.random.choice(sparsity), replace=False)) for _ in range(n_samples)]

    Lt = [pts[ind] for ind in ind_each]
    Ly = [samp[i, ind] for i, ind in enumerate(ind_each)]
    
    return {'Lt': Lt, 'Ly': Ly}

def remote_sampling(N, s):
    """
    Perform aggressive sparsification by ensuring minimum spacing between sampled points.
    
    Parameters:
    - N: int - total number of possible points
    - s: list - possible number of points to sample
    
    Returns:
    - np.ndarray - sorted array of sampled indices
    """
    onesamp = np.sort(np.random.choice(N, np.random.choice(s), replace=False))
    threshold = (1 / len(onesamp))**1.5 * N
    while np.min(np.diff(onesamp)) < threshold:
        onesamp = np.sort(np.random.choice(N, len(onesamp), replace=False))
    return onesamp
