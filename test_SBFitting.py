import numpy as np
from scipy.stats import norm
import unittest
from SBFitting import SBFitting

# Assuming SBFitting is already implemented
# from your_module import SBFitting

class TestSBFitting(unittest.TestCase):
    
    def test_algorithm_convergence_and_estimation_precision(self):
        np.random.seed(100)
        n = 100
        d = 2
        
        # Generate X using a similar process as in R
        X = norm.cdf(np.random.randn(n, d) @ np.array([[1, 0.6], [0.6, 1]]))
        
        # Define f1 and f2
        def f1(t):
            return 2 * (t - 0.5)
        
        def f2(t):
            return np.sin(2 * np.pi * t)
        
        # Generate Y
        Y = f1(X[:, 0]) + f2(X[:, 1]) + np.random.normal(0, 0.1, n)
        
        # Define N and x
        N = 101
        x = np.tile(np.linspace(0, 1, N), (d, 1)).T
        
        # Define h
        h = np.array([0.12, 0.08])
        
        # Call the SBFitting function (assuming it's implemented)
        sbf_result = SBFitting(Y, x, X, h)
        fFit = sbf_result['SBFit']
        
        # Retrieve errors and iteration parameters
        iterErr = sbf_result['iterErr']
        iterErrDiff = sbf_result['iterErrDiff']
        iterNum = sbf_result['iterNum']
        critErr = sbf_result['critErr']
        critErrDiff = sbf_result['critErrDiff']
        critNum = sbf_result['critNum']
        
        # Tests
        self.assertTrue(np.sum(fFit.shape != (N, d)) == 0)
        self.assertTrue((iterErr < critErr) + (iterErrDiff < critErrDiff) + (iterNum < critNum) < 3)
        
        estErr = np.array([np.max(np.abs(f1(x[:, 0]) - fFit[:, 0])), np.max(np.abs(f2(x[:, 1]) - fFit[:, 1]))])
        
        crit1 = np.sqrt(np.log(n) / n / h**2)
        crit2 = h
        
        crit = np.maximum(crit1, crit2)
        
        self.assertTrue(estErr[0] / crit[0] < 1)
        self.assertTrue(estErr[1] / crit[1] < 1)
    
    def test_algorithm_convergence_and_prediction_performance(self):
        np.random.seed(100)
        n = 100
        d = 2
        
        # Generate X using a similar process as in R
        X = norm.cdf(np.random.randn(n, d) @ np.array([[1, 0.6], [0.6, 1]]))
        
        # Define f1 and f2
        def f1(t):
            return 2 * (t - 0.5)
        
        def f2(t):
            return np.sin(2 * np.pi * t)
        
        # Generate Y
        Y = f1(X[:, 0]) + f2(X[:, 1]) + np.random.normal(0, 0.1, n)
        
        # Define N and x
        N = n
        x = X
        
        # Define h
        h = np.array([0.12, 0.08])
        
        # Call the SBFitting function (assuming it's implemented)
        sbf_result = SBFitting(Y, x, X, h)
        fFit = sbf_result['SBFit']
        
        # Retrieve errors and iteration parameters
        iterErr = sbf_result['iterErr']
        iterErrDiff = sbf_result['iterErrDiff']
        iterNum = sbf_result['iterNum']
        critErr = sbf_result['critErr']
        critErrDiff = sbf_result['critErrDiff']
        critNum = sbf_result['critNum']
        
        # Tests
        self.assertTrue(np.sum(fFit.shape != (N, d)) == 0)
        self.assertTrue((iterErr < critErr) + (iterErrDiff < critErrDiff) + (iterNum < critNum) < 3)
        
        # Compute mean squared errors
        mY = sbf_result['mY']
        mseSBF = np.mean((Y - mY - np.sum(fFit, axis=1))**2)
        
        mseConst = np.var(Y)
        mseLM = np.mean((np.linalg.lstsq(X, Y, rcond=None)[1])**2)
        
        self.assertTrue(mseSBF < mseConst)
        self.assertTrue(mseSBF < mseLM)

if __name__ == '__main__':
    unittest.main()
