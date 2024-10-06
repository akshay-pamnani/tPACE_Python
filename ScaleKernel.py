import numpy as np

# Uniform distribution support function
def dunif(x, lower, upper):
    return np.where((x >= lower) & (x <= upper), 1/(upper - lower), 0)

# Assuming dunif is already defined
def ScaleKernel(x, X, h=None, K='epan', supp=None):
    N = len(x)
    n = len(X)

    if K != 'epan':
        print('Epanechnikov kernel is only supported currently. It uses Epanechnikov kernel automatically.')

    if supp is None:
        supp = np.array([0, 1])
    
    if h is None:
        h = 0.25 * n**(-1/5) * (supp[1] - supp[0])

    # Create matrices for broadcasting
    xTmp = np.tile(x[:, np.newaxis], (1, n))  # shape (N, n)
    XTmp = np.tile(X[np.newaxis, :], (N, 1))  # shape (N, n)

    Tmp = xTmp - XTmp

    KhTmp = (3/4) * (1 - (Tmp/h)**2) * dunif(Tmp/h, -1, 1) * (2/h)

    return KhTmp
