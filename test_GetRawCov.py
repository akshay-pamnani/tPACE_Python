import numpy as np
import unittest
import rdata
from scipy.interpolate import interp1d
from GetRawCov import GetRawCov
from SetOptions import set_options

# Assuming the GetRawCov function has been defined earlier as discussed
# Replace with your actual GetRawCov function here


class TestGetRawCov(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the data from the RData file
        parsed = rdata.parser.parse_file("dataForGetRawCov.RData")
        data = rdata.conversion.convert(parsed)

        print("Keys in the RData file:", data.keys())

        # Assuming y and t are arrays saved in the RData file
        cls.y = data['y']  # assuming y is saved as a numpy array
        cls.t = data['t']  # assuming t is saved as a numpy array
        cls.mu = data['mu'] 

        # Set options with the default kernel (epanechnikov kernel)
        p = {'kernel': 'epan'}
        cls.optns = set_options(cls.y, cls.t, p)

        # Generate grids for observation and regular grids
        cls.out1 = sorted(set(np.concatenate(cls.t)))  # Observation grid
        cls.out21 = np.linspace(min(cls.out1), max(cls.out1), num=30)  # Regular grid

    def test_get_raw_cov_sparse_true(self):
        # Equivalent to GetRawCov(y, t, sort(unlist(t)), mu, 'Sparse', TRUE)
        AA = GetRawCov(self.y, self.t, np.sort(np.unique(np.concatenate(self.t))), self.mu, 'Sparse', True)

        print("y",self.y)
        print("t",self.t)
        print("mu",self.mu)

        unique_t = np.sort(np.unique(np.concatenate(self.t)))
        print("Unique sorted t values:", unique_t)
        print("mu values:", self.mu)

        print("Values of indx:", AA.get('indx'))
        print("Sum of indx:", np.sum(AA.get('indx')))
        print("Size of cyy:", len(AA['cyy']))

        


        # Test cases for Sparse, True
        assert np.isclose(np.sum(AA['indx']), 184, atol=2 * np.finfo(float).eps)
        assert np.isclose(np.sum(AA['cxxn']), -7.416002855888680, atol=1e-13)
        assert np.isclose(np.sum(AA['cyy']), 16.327874649330514, atol=1e-13)
        assert np.isclose(np.sum(AA['tPairs']), 4.053285461728229e+02, atol=1e-12)

    def test_get_raw_cov_sparse_false(self):
        # Equivalent to GetRawCov(y, t, sort(unlist(t)), mu, 'Sparse', FALSE)
        BB = GetRawCov(self.y, self.t, np.sort(np.unique(np.concatenate(self.t))), self.mu, 'Sparse', False)

        # Test cases for Sparse, False
        assert np.isclose(np.sum(BB['indx']), 298, atol=2 * np.finfo(float).eps)
        assert np.isclose(np.sum(BB['cxxn']), 16.327874649330514, atol=1e-13)
        assert np.isclose(np.sum(BB['cyy']), 16.327874649330514, atol=1e-13)
        assert np.isclose(np.sum(BB['tPairs']), 6.330209554605514e+02, atol=1e-12)

    def test_get_raw_cov_dense_true():
        # Example for Dense data
        y2 = [list(range(1, 11)), list(range(2, 12))]
        t2 = [list(range(1, 11)), list(range(1, 11))]
        mu2 = np.linspace(1.5, 10.5, 10)  # Sample mu
        CC = GetRawCov(y2, t2, np.sort(np.unique(np.concatenate(t2))), mu2, 'Dense', True)

        # Test cases for Dense, True
        assert np.isclose(np.sum(CC['indx']), 0, atol=2 * np.finfo(float).eps)
        assert np.isclose(np.sum(CC['cxxn']), 22.5, atol=1e-13)
        assert np.isclose(np.sum(CC['cyy']), 25, atol=1e-13)
        assert np.isclose(np.sum(CC['tPairs']), 990, atol=1e-12)

    def test_get_raw_cov_dense_false():

        y2 = [list(range(1, 11)), list(range(2, 12))]
        t2 = [list(range(1, 11)), list(range(1, 11))]
        mu2 = np.linspace(1.5, 10.5, 10)  # Sample mu

        # Example for Dense data, False
        DD = GetRawCov(y2, t2, np.sort(np.unique(np.concatenate(t2))), mu2, 'Dense', False)

        # Test cases for Dense, False
        assert np.isclose(np.sum(DD['indx']), 0, atol=2 * np.finfo(float).eps)
        assert np.isclose(np.sum(DD['cxxn']), 25, atol=1e-13)
        assert np.isclose(np.sum(DD['cyy']), 25, atol=1e-13)
        assert np.isclose(np.sum(DD['tPairs']), 1100, atol=1e-12)

# To run the tests
if __name__ == "__main__":
    unittest.main()
