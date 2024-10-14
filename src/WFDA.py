import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.optimize import minimize
import time
from CheckData import CheckData
from CheckOptions import check_options
from List2Mat import list_to_mat
from HandleNumericsAndNan import handle_numerics_and_nan
from GetMeanDense import get_mean_dense
from SetOptions import set_options
import sys
import os
sys.path.append(os.path.abspath('src'))
from trapzRcpp import trapz
from RcppPseudoApprox import pseudo_approx
from Rcppsort import pybind_sort

def WFDA(Ly, Lt, optns=None):
    if optns is None:
        optns = {}
    
    # Set default options
    optns.setdefault('isPWL', True)
    optns.setdefault('verbose', False)
    optns.setdefault('nknots', 2 if optns['isPWL'] else None)
    optns.setdefault('subsetProp', 0.50)
    optns.setdefault('choice', 'truncated')
    optns.setdefault('seed', 666)
    
    # Validate options
    if optns['isPWL'] and (not (1 <= optns['nknots'] <= 7)):
        raise ValueError("Number of knots should be between 1 and 7.")
    if not (0 < optns['subsetProp'] <= 1):
        raise ValueError("Number of proportion used should be above 0 and at most 1.")
    if optns['choice'] not in ['truncated', 'weighted']:
        raise ValueError("The estimation of warping functions can only be done by 'truncated' or 'weighted' average.")
    
    # Handle randomness
    np.random.seed(optns['seed'])

    # Assume CheckData and HandleNumericsAndNAN functions process data
    CheckData(Ly,Lt) 
    inputData = handle_numerics_and_nan(Ly, Lt)
    Ly = inputData['Ly']
    Lt = inputData['Lt']

    # Check data is dense
    optnsFPCA = set_options(Ly, Lt, {})
    if optnsFPCA['dataType'] != 'Dense':
        raise ValueError(f"The data has to be 'Dense' for WFDA to be relevant; the current dataType is : '{optnsFPCA['dataType']}'!")

    numOfCurves = len(Ly)
    check_options(Lt, optnsFPCA, numOfCurves)

    ymat = list_to_mat(Ly, Lt)
    N = ymat.shape[0]

    obsGrid = np.sort(np.unique(np.hstack(Lt)))
    workGrid = (obsGrid - obsGrid.min()) / (obsGrid.max() - obsGrid.min())
    M = len(workGrid)

    # Super-norm normalization
    maxAtTimeT = np.max(np.abs(ymat), axis=1)
    ymatNormalised = ymat / maxAtTimeT[:, None]

    # Mean function
    smcObj = get_mean_dense(ymatNormalised, obsGrid, optnsFPCA)
    mu = smcObj['mu']

    # Penalty parameter
    if 'lambda' not in optns:
        Vy = np.sqrt(np.sum([simps((ymatNormalised[i] - mu) ** 2, obsGrid) for i in range(N)]) / (N - 1))
        lambda_ = Vy * 1e-4
    else:
        lambda_ = optns['lambda']

    if optns['lambda'] is None:
        Vy = np.sqrt(np.sum(np.apply_along_axis(lambda u: trapz(obsGrid, (u - mu) ** 2), axis=1, arr=ymatNormalised)) / (N - 1))
        lambda_ = Vy * 10 ** -4
    else:
        lambda_ = optns['lambda']

    numOfKcurves = min(round(optns['subsetProp'] * (N - 1)))
    hikMat = np.empty((numOfKcurves, M, N))
    distMat = np.empty((N, numOfKcurves))
    hMat = np.empty((N, M))
    hInvMat = np.empty((N, M))
    alignedMat = np.empty((N, M))
    
    def getcurveJ(tj, curvej):
        return pseudo_approx(workGrid, curvej, tj)

    def theCost(curvei, curvek, lambda_, ti, tk):
        return np.sum((getcurveJ(tk, curvek) - curvei) ** 2) + lambda_ * np.sum((tk - ti) ** 2)

    def get_hikrs(curvei, curvek, lambda_param):
        my_costs = []

        for u in range(1, numOfKcurves ** 2 + 1):
            np.random.seed(u)  # Set seed for reproducibility
            random_values = np.random.rand(m - 2)
            cost = theCost(curvei, curvek, lambda_, workGrid, np.concatenate(([0], pybind_sort(random_values), [1])))
            my_costs.append(cost)

        np.random.seed(np.argmin(my_costs) + 1)  # Set seed based on minimum cost index
        # min_cost = np.min(my_costs)
        
        return np.concatenate(([0], pybind_sort(np.random.rand(m - 2)), [1]))

    def get_sol(x):
        sorted_x = pybind_sort(x)  # Sort using rcppsort
        y = np.concatenate(([0], sorted_x, [1]))
        x_seq = np.linspace(0, 1, 2 + optns['nknots'])
        return pseudo_approx(x_seq, y, np.linspace(0, 1, M))
    
    def theCostOptim(x, curvei, curvek, lambda_, ti):
        tk = get_sol(x)
        return np.sum((getcurveJ(tk, curvek) - curvei) ** 2) + lambda_ * np.sum((tk - ti) ** 2)

    def get_hik_optim(curvei, curvek, lambda_param, minqa_avail, optns):
        s0 = np.linspace(0, 1, 2 + optns['nknots'])[1:(1 + optns['nknots'])]  # Initial parameters
        bounds = [(1e-6, 1 - 1e-6)] * optns['nknots']  # Bounds for the parameters

        if not minqa_avail:
            optim_res = minimize(
                theCostOptim, 
                s0, 
                args=(curvei, curvek, lambda_param, workGrid), 
                method='L-BFGS-B', 
                bounds=bounds
            )
        else:
            # Using scipy's optimization, adjust this if you have a specific Bobyqa implementation
            # You may need to find an alternative for the Bobyqa method
            optim_res = minimize(
                theCostOptim,
                s0,
                args=(curvei, curvek, lambda_param, workGrid),
                method='trust-constr',  # Example; adjust as necessary for equivalent behavior
                bounds=bounds
            )

        best_sol = get_sol(optim_res.x)  # Adjust M as needed or pass it directly
        return best_sol

    start_time = time.time()
    
    # Check for minqa availability
    minqa_avail = 'minqa' in sys.modules or optns['isPWL']  # Placeholder for minqa check
    if not minqa_avail:
        print("Warning: Cannot use 'minqa::bobyqa' to find the optimal knot locations as 'minqa' is not installed. We will do an 'L-BFGS-B' search.")
    
    N = ymatNormalised.shape[0]  # Number of curves
    # hik_mat = np.zeros((numOfKcurves, M, N))  # Replace M with the appropriate dimension
    # dist_mat = np.zeros((N, numOfKcurves))
    # h_inv_mat = np.zeros((N, M))  # Adjust dimensions as necessary
    # h_mat = np.zeros((N, M))  # Adjust dimensions as necessary
    # aligned_mat = np.zeros((N, M))  # Adjust dimensions as necessary

    for i in range(N):  # For each curve
        if optns['verbose']:
            print(f'Computing pairwise warping for curve #: {i + 1} out of {N} curves.')
        
        np.random.seed(i + optns['seed'])
        curvei = ymatNormalised[i, :]
        candidate_kcurves = np.random.choice(np.delete(np.arange(N), i), numOfKcurves, replace=False)

        for k in range(numOfKcurves):  # For each candidate curve
            if not optns['isPWL']:
                hikMat[k, :, i] = get_hikrs(curvei, ymatNormalised[candidate_kcurves[k], :], lambda_)
            else:
                hikMat[k, :, i] = get_hik_optim(curvei, ymatNormalised[candidate_kcurves[k], :], lambda_, minqa_avail, optns)
            
            # Calculate the distance
            distMat[i, k] = np.mean((getcurveJ(tj=hikMat[k, :, i], curvej=curvei) - ymatNormalised[candidate_kcurves[k]]) ** 2)

        if optns['choice'] == 'weighted':
            hInvMat[i, :] = np.average(hikMat[:, :, i], axis=0, weights=1/distMat[i, :])
        else:
            hInvMat[i, :] = np.mean(hikMat[distMat[i, :] <= np.quantile(distMat[i, :], 0.90), :, i], axis=0)

        # Interpolate h_mat and aligned_mat
        hMat[i, :] = interp1d(workGrid, hInvMat[i, :], bounds_error=False)(workGrid)
        alignedMat[i, :] = interp1d(workGrid, ymatNormalised[i, :], bounds_error=False)(hMat[i, :])

    timing = time.time() - start_time
    ret = {
        'optns': optns,
        'lambda': lambda_,
        'h': hMat,
        'hInv': hInvMat,
        'aligned': alignedMat,
        'costs': np.mean(distMat, axis=1),
        'timing': timing
    }
    
    # Set the class, if needed
    # If you want to handle class-like structures, consider using a custom class
    return ret
