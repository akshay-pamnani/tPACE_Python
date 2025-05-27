import numpy as np
from scipy.interpolate import interp1d
from ConvertSupport import convert_support
from typing import List, Dict, Any

def get_ce_scores(y: List[np.ndarray], t: List[np.ndarray], optns: Dict[str, Any],
                  mu: np.ndarray, obs_grid: np.ndarray, fitted_cov: np.ndarray,
                  lambda_: np.ndarray, phi: np.ndarray, sigma2: float = 0.0) -> List[Dict[str, Any]]:
    if lambda_.shape[0] != phi.shape[1]:
        raise ValueError("Number of eigenvalues does not match number of eigenfunctions.")
    
    sigma_y = fitted_cov + np.eye(phi.shape[0]) * sigma2
    mu_phi_sig = get_mu_phi_sig(t, obs_grid, mu, phi, sigma_y)
    
    results = []
    for y_vec, mps in zip(y, mu_phi_sig):
        result = get_ind_ce_scores(y_vec, mps['mu_vec'], lambda_, mps['phi_mat'], mps['sigma_yi'],
                                   verbose=optns.get('verbose', False))
        results.append(result)
    return results

def get_mu_phi_sig(t: List[np.ndarray], obs_grid: np.ndarray, mu: np.ndarray,
                   phi: np.ndarray, sigma_y: np.ndarray) -> List[Dict[str, Any]]:
    mu_interp = interp1d(obs_grid, mu, kind='linear', fill_value="extrapolate")
    phi_interps = [
        interp1d(obs_grid, phi[:, i], kind='linear', fill_value="extrapolate")
        for i in range(phi.shape[1])
    ]

    ret = []
    for tvec in t:
        if len(tvec) == 0:
            ret.append({'mu_vec': np.array([]), 'phi_mat': np.array([]), 'sigma_yi': np.array([])})
            continue
        
        mu_vec = mu_interp(tvec)
        phi_mat = np.column_stack([interp(tvec) for interp in phi_interps])
        sigma_yi = convert_support(obs_grid, tvec, mu=sigma_y)
        
        ret.append({'mu_vec': mu_vec, 'phi_mat': phi_mat, 'sigma_yi': sigma_yi})
    
    return ret



def get_ind_ce_scores(y_vec: np.ndarray, mu_vec: np.ndarray, lam_vec: np.ndarray,
                      phi_mat: np.ndarray, sigma_yi: np.ndarray,
                      newy_ind: int = None, verbose: bool = False) -> Dict[str, Any]:
    if len(y_vec) == 0:
        if verbose:
            print("Empty observation found, possibly due to truncation.")
        return {
            'xi_est': np.full((len(lam_vec),), np.nan),
            'xi_var': np.full((len(lam_vec), len(lam_vec)), np.nan),
            'fitted_y': np.full((0, 0), np.nan)
        }
    
    if newy_ind is not None:
        if len(y_vec) != 1:
            new_phi = phi_mat[newy_ind, :].reshape(1, -1)
            new_mu = mu_vec[newy_ind]
            y_vec = np.delete(y_vec, newy_ind)
            mu_vec = np.delete(mu_vec, newy_ind)
            phi_mat = np.delete(phi_mat, newy_ind, axis=0)
            sigma_yi = np.delete(np.delete(sigma_yi, newy_ind, axis=0), newy_ind, axis=1)
            return GetIndCEScoresCPPnewInd(y_vec, mu_vec, lam_vec, phi_mat, sigma_yi, new_phi, new_mu)
        else:
            lam_phi = np.diag(lam_vec) @ phi_mat.T
            lam_phi_sig = lam_phi @ np.linalg.inv(sigma_yi)
            xi_est = lam_phi_sig @ (y_vec - mu_vec)
            xi_var = np.diag(lam_vec) - lam_phi @ lam_phi_sig.T
            return {'xi_est': xi_est, 'xi_var': xi_var, 'fitted_y': np.nan}
    
    return GetIndCEScoresCPP(y_vec, mu_vec, lam_vec, phi_mat, sigma_yi)
