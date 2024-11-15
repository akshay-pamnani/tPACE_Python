import numpy as np
from Sparsify import sparsify_sample

def wiener(n=1, pts=np.linspace(0, 1, 50), sparsify=None, K=50):
    """
    Simulate n standard Wiener processes on the interval [0, 1], with optional sparsification.
    
    Parameters:
    - n: int - number of samples
    - pts: np.ndarray - vector of points in [0, 1] specifying the support of the processes
    - sparsify: list or None - list of integers specifying the number of observations per curve
    - K: int - number of components
    
    Returns:
    - np.ndarray - matrix of samples with n rows. If sparsify is specified, returns the sparsified sample.
    """
    
    # Ensure pts is a column vector
    pts = np.atleast_2d(pts).T if pts.ndim == 1 else pts
    
    # Create the basis matrix using Karhunen-Loève expansion
    basis = np.sqrt(2) * np.sin(pts @ (np.arange(1, K + 1) - 0.5) * np.pi)
    
    # Generate samples
    samp = (basis @ np.diag(1 / (np.arange(1, K + 1) - 0.5) / np.pi) @ np.random.randn(K, n)).T
    
    # Apply sparsification if specified
    if sparsify is not None:
        samp = sparsify_sample(samp, pts, sparsify)
        
    return samp

