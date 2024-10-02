import unittest
import numpy as np
import rdata
from scipy.interpolate import interp1d

# Import the required functions
from GetSmoothenedMeanCurve import get_smoothed_mean_curve
from SetOptions import set_options

class TestSmoothedMeanCurve(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the data from the RData file
        parsed = rdata.parser.parse_file("dataGeneratedByExampleSeed123.RData")
        data = rdata.conversion.convert(parsed)

        # Assuming y and t are arrays saved in the RData file
        cls.y = data['y']  # assuming y is saved as a numpy array
        cls.t = data['t']  # assuming t is saved as a numpy array

        # Set options with the default kernel (epanechnikov kernel)
        p = {'kernel': 'epan'}
        cls.optns = set_options(cls.y, cls.t, p)

        # Generate grids for observation and regular grids
        cls.out1 = sorted(set(np.concatenate(cls.t)))  # Observation grid
        cls.out21 = np.linspace(min(cls.out1), max(cls.out1), num=30)  # Regular grid

    def test_epan_kernel(self):
        # Test with the Epanechnikov kernel
        smc_obj = get_smoothed_mean_curve(y=self.y, t=self.t, obs_grid=self.out1, reg_grid=self.out21, optns=self.optns)
        # Assert that the sum of the smoothed curve is close to the expected value
        self.assertAlmostEqual(np.sum(smc_obj['mu']), 1.176558873333339e+02, delta=4)

    def test_rect_kernel(self):
        # Test with the rectangular kernel
        self.optns['kernel'] = 'rect'
        smc_obj = get_smoothed_mean_curve(y=self.y, t=self.t, obs_grid=self.out1, reg_grid=self.out21, optns=self.optns)
        self.assertAlmostEqual(np.sum(smc_obj['mu']), 1.186398254457767e+02, delta=6)

    def test_gauss_kernel(self):
        # Test with the Gaussian kernel
        self.optns['kernel'] = 'gauss'
        smc_obj = get_smoothed_mean_curve(y=self.y, t=self.t, obs_grid=self.out1, reg_grid=self.out21, optns=self.optns)
        self.assertAlmostEqual(np.sum(smc_obj['mu']), 1.206167514696777e+02, delta=4)

if __name__ == '__main__':
    unittest.main()
