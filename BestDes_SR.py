import numpy as np
from itertools import combinations

def src_criteria(design, ridge, Cov, CCov):
    # Optimization criterion for SR
    design = sorted(design)
    ridgeCov = Cov + np.diag(ridge)
    srcri = CCov[design].T @ np.linalg.inv(ridgeCov[np.ix_(design, design)]) @ CCov[design]
    return srcri

def best_des_sr(p, ridge, workGrid, Cov, CCov, is_sequential=False):
    # select optimal designs for regression case, sequential method available
    if not is_sequential:
        comb_list = list(combinations(range(len(workGrid)), p))
        temps = np.array([src_criteria(comb, ridge, Cov, CCov) for comb in comb_list])
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
                seqcri[idx] = src_criteria(temp_des, ridge, Cov, CCov)
            opt_des.append(candidx[np.argmax(seqcri)])
        return {'best': sorted(opt_des), 'med': None}
