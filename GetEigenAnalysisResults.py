import numpy as np
import sys
import os
sys.path.append(os.path.abspath('src'))
from trapzRcpp import trapz


def get_eigen_analysis_results(smoothCov, regGrid, optns, muWork=None):
    """
    Perform eigenanalysis on the covariance matrix and select components
    based on specified variance explained threshold.
    
    Parameters:
    - smoothCov: np.ndarray - covariance matrix
    - regGrid: np.ndarray - regular grid for integration
    - optns: dict - options containing:
        * 'maxK': int, maximum number of principal components
        * 'FVEthreshold': float, functional variance explained threshold
        * 'FVEfittedCov': float, threshold for fitted covariance, if applicable
        * 'verbose': bool, whether to print messages
    - muWork: np.ndarray or None, optional mean work vector (default None)

    Returns:
    - dict with keys:
        * 'lambda': np.ndarray - eigenvalues selected
        * 'phi': np.ndarray - selected eigenvectors
        * 'cumFVE': np.ndarray - cumulative FVE
        * 'kChoosen': int - number of components chosen
        * 'fittedCov': np.ndarray - fitted covariance
        * 'fittedCovUser': np.ndarray or None - fitted covariance with user-specified threshold
        * 'fittedCorrUser': np.ndarray or None - correlation matrix if diagonal is non-zero
    """
    maxK = optns['maxK']
    FVEthreshold = optns['FVEthreshold']
    FVEfittedCov = optns.get('FVEfittedCov', None)
    verbose = optns['verbose']
    
    gridSize = regGrid[1] - regGrid[0]
    numGrids = smoothCov.shape[0]
    
    # Eigen decomposition
    eig_values, eig_vectors = np.linalg.eigh(smoothCov)
    
    # Select positive eigenvalues
    positive_ind = eig_values >= 0
    if np.sum(positive_ind) == 0:
        raise ValueError("All eigenvalues are negative. The covariance estimate is incorrect.")
    
    d = eig_values[positive_ind][::-1]  # Sort in descending order
    eigenV = eig_vectors[:, positive_ind][:, ::-1]  # Match ordering with eigenvalues

    # Threshold based on maxK
    if maxK < len(d):
        if verbose:
            print(f"At most {len(d)} number of PCs can be selected, thresholded by `maxK` = {maxK}.")
        
        d = d[:maxK]
        eigenV = eigenV[:, :maxK]

    # Calculate cumulative FVE
    FVE = np.cumsum(d) / np.sum(d)
    no_opt = np.min(np.where(FVE >= FVEthreshold)[0]) + 1  # Select minimum components for FVE threshold
    
    # Normalization of eigenvectors
    if muWork is None:
        muWork = np.arange(eigenV.shape[0]) + 1  # Default mean work

    def normalize_vector(x):
        """Normalize vector x using trapezoidal integration and adjust sign based on mean."""
        x /= np.sqrt(trapz(regGrid, x**2))
        return x if np.sum(x * muWork) >= 0 else -x
    
    phi = np.apply_along_axis(normalize_vector, 0, eigenV)
    lambda_ = gridSize * d
    
    # Covariance matrix construction
    no_fittedCov = np.min(np.where(FVE >= FVEfittedCov)[0]) + 1 if FVEfittedCov is not None else phi.shape[1]
    fittedCovUser = phi[:, :no_fittedCov] @ np.diag(lambda_[:no_fittedCov]) @ phi[:, :no_fittedCov].T
    fittedCov = phi @ np.diag(lambda_) @ phi.T

    # Fitted correlation matrix
    if np.any(np.diag(fittedCovUser) == 0):
        fittedCorrUser = None
    else:
        diag_sqrt_inv = np.diag(1 / np.sqrt(np.diag(fittedCovUser)))
        fittedCorrUser = diag_sqrt_inv @ fittedCovUser @ diag_sqrt_inv
        np.fill_diagonal(fittedCorrUser, 1)

    return {
        'lambda': lambda_[:no_opt],
        'phi': phi[:, :no_opt],
        'cumFVE': FVE,
        'kChoosen': no_opt,
        'fittedCov': fittedCov,
        'fittedCovUser': fittedCovUser,
        'fittedCorrUser': fittedCorrUser
    }