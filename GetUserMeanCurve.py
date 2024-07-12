import numpy as np
from scipy.interpolate import interp1d

class SMC:
    def __init__(self, mu, muDense, bw_mu):
        self.mu = mu
        self.muDense = muDense
        self.bw_mu = bw_mu

def GetUserMeanCurve(optns, obsGrid, regGrid, buff):
    # For the case where a user provides a mean function

    userMu = optns['userMu']
    rangeUser = (min(userMu['t']), max(userMu['t']))
    rangeObs = (min(obsGrid), max(obsGrid))
    
    if rangeUser[0] > rangeObs[0] + buff or rangeUser[1] < rangeObs[1] - buff:
        raise ValueError('The range defined by the user provided mean does not cover the support of the data.')

    spline_func = interp1d(userMu['t'], userMu['mu'], kind='cubic', fill_value="extrapolate")
    mu = spline_func(obsGrid)
    spline_func_dense = interp1d(obsGrid, mu, kind='cubic', fill_value="extrapolate")
    muDense = spline_func_dense(regGrid)
    bw_mu = None

    result = SMC(mu=mu, muDense=muDense, bw_mu=bw_mu)
    return result
