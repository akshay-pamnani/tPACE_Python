import numpy as np
import sys
import os
sys.path.append(os.path.abspath('src'))
from CPPlwls1d_py import CPPlwls1d

def lwls_1d(bw, kernel_type, win=None, xin=None, yin=None, xout=None, npoly=1, nder=0):
    if win is None:
        win = np.ones_like(xin)
    if xin is None or yin is None or xout is None:
        raise ValueError('xin, yin, and xout must be provided.')
    
    if not np.all(np.diff(xin) > 0):
        raise ValueError('`xin` needs to be sorted in increasing order')
    
    if not np.all(np.diff(xout) > 0):
        raise ValueError('`xout` needs to be sorted in increasing order')
    
    if np.all(np.isnan(win)) or np.all(np.isnan(xin)) or np.all(np.isnan(yin)):
        raise ValueError('win, xin or yin contain only NAs!')

    # Deal with NA/NaN measurement values
    mask = ~np.isnan(xin) & ~np.isnan(yin) & ~np.isnan(win)
    win = win[mask]
    xin = xin[mask]
    yin = yin[mask]

    # Assuming CPPlwls1d is a function defined in your C++ extension or similar
    return CPPlwls1d(bw=float(bw), kernel_type=kernel_type, npoly=int(npoly), 
                      nder=int(nder), xin=xin.astype(float), 
                      yin=yin.astype(float), xout=xout.astype(float), 
                      win=win.astype(float))

