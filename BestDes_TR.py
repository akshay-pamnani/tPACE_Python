import numpy as np
from scipy.integrate import trapz
from itertools import combinations

def trcri(design, ridge, Cov, workGrid):
    # Optimization criterion for TR
    design = sorted(design)
    RidgeCov = Cov + np.diag(ridge)
    design_cov_inv = np.linalg.inv(RidgeCov[np.ix_(design, design)])
    
    if len(design) > 1:
        diag_term = np.diag(Cov[design, :] @ design_cov_inv @ Cov[design, :].T)
    else:
        diag_term = np.diag(Cov[design, :] @ design_cov_inv @ Cov[design, :].T)
    
    trcri_value = trapz(diag_term, x=workGrid)
    return trcri_value

def best_des_tr(p, ridge, workGrid, Cov, is_sequential=False):
    # Select optimal designs for trajectory recovery case, sequential method available
    if not is_sequential:
        comb_list = list(combinations(range(len(workGrid)), p))
        temps = np.array([trcri(comb, ridge, Cov, workGrid) for comb in comb_list])
        best_idx = np.argmax(temps)
        best = sorted(comb_list[best_idx])
        return {'best': best}
    else:
        opt_des = []
        for _ in range(p):
            candidx = [i for i in range(len(workGrid)) if i not in opt_des]
            seqcri = np.zeros(len(candidx))
            for idx, candidate in enumerate(candidx):
                temp_des = sorted(opt_des + [candidate])
                seqcri[idx] = trcri(temp_des, ridge, Cov, workGrid)
            opt_des.append(candidx[np.argmax(seqcri)])
        return {'best': sorted(opt_des), 'med': None}
