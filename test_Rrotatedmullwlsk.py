import numpy as np
import pytest
import rdata
import sys
import os
sys.path.append(os.path.abspath('src'))
from Rrotatedmullwlsk import rotatedmullwlsk         # to load the RData-converted input

# Load input
# Load the data from the RData file
parsed = rdata.parser.parse_file("InputForRotatedMllwlskInCpp.RData")
data1 = rdata.conversion.convert(parsed)
IN = data1

TOL = 1e-13

def test_epan_kernel():
    AA = rotatedmullwlsk(bw=IN['bw'], tPairs=IN['tPairs'], cxxn=IN['cxxn'], win=IN['win'],
                          xygrid=IN['xygrid'], npoly=IN['npoly'], kernel_type=IN['kernel'], bwCheck=False)
    BB = rotatedmullwlsk(bw=[3, 4], tPairs=IN['tPairs'], cxxn=IN['cxxn'], win=IN['win'],
                          xygrid=IN['xygrid'], npoly=IN['npoly'], kernel_type=IN['kernel'], bwCheck=False)
    CC = rotatedmullwlsk(bw=[13, 23.3], tPairs=IN['tPairs'], cxxn=IN['cxxn'], win=IN['win'],
                          xygrid=IN['xygrid'], npoly=IN['npoly'], kernel_type=IN['kernel'], bwCheck=False)

    np.testing.assert_allclose(np.sum(AA), -1.887451898050793, rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.sum(BB), -3.264859562745997, rtol=0, atol=1e-11)
    np.testing.assert_allclose(np.sum(CC), -5.650324984396344, rtol=0, atol=1e-13)

def test_rect_kernel():
    for bw, expected in [([IN['bw']], 0.408929466844517),
                         ([3, 4], -1.803538175275243),
                         ([13, 23.3], -5.866207150638594)]:
        result = rotatedmullwlsk(bw=bw, tPairs=IN['tPairs'], cxxn=IN['cxxn'], win=IN['win'],
                                  xygrid=IN['xygrid'], npoly=IN['npoly'], kernel_type='rect', bwCheck=False)
        np.testing.assert_allclose(np.sum(result), expected, rtol=0, atol=TOL)

def test_gaussian_kernel():
    for bw, expected, tol in [([IN['bw']], -4.197686977022681, 1e-13),
                              ([3, 4], -4.134314374205185, 1e-14),
                              ([13, 23.3], -5.767647736432502, 1e-13)]:
        result = rotatedmullwlsk(bw=bw, tPairs=IN['tPairs'], cxxn=IN['cxxn'], win=IN['win'],
                                  xygrid=IN['xygrid'], npoly=IN['npoly'], kernel_type='gauss', bwCheck=False)
        np.testing.assert_allclose(np.sum(result), expected, rtol=0, atol=tol)

def test_quartic_kernel():
    for bw, expected in [([IN['bw']], -3.753442160580053),
                         ([3, 4], -4.970567279909929),
                         ([13, 23.3], -5.443792883622939)]:
        result = rotatedmullwlsk(bw=bw, tPairs=IN['tPairs'], cxxn=IN['cxxn'], win=IN['win'],
                                  xygrid=IN['xygrid'], npoly=IN['npoly'], kernel_type='quar', bwCheck=False)
        np.testing.assert_allclose(np.sum(result), expected, rtol=0, atol=TOL)

def test_gausvar_kernel():
    for bw, expected in [([IN['bw']], -9.228691155965564),
                         ([3, 4], -3.594812776733668),
                         ([13, 23.3], -5.718225024334538)]:
        result = rotatedmullwlsk(bw=bw, tPairs=IN['tPairs'], cxxn=IN['cxxn'], win=IN['win'],
                                  xygrid=IN['xygrid'], npoly=IN['npoly'], kernel_type='gausvar', bwCheck=False)
        np.testing.assert_allclose(np.sum(result), expected, rtol=0, atol=TOL)

def test_positive_weights():
    win1 = np.arange(1, 39)
    win2 = np.sin(np.arange(1, 39)) + 3

    kernels = [
        ('gausvar', win1, -4.924560108566402),
        ('gauss', win1, -6.577000474589042),
        ('rect', win1, -1.791956888763226),
        ('epan', win2, -3.614424355861832),
        ('quar', win2, -5.450343839504677),
    ]

    for kernel, win, expected in kernels:
        result = rotatedmullwlsk(bw=[3, 4], tPairs=IN['tPairs'], cxxn=IN['cxxn'], win=win,
                                  xygrid=IN['xygrid'], npoly=IN['npoly'], kernel_type=kernel, bwCheck=False)
        np.testing.assert_allclose(np.sum(result), expected, rtol=0, atol=TOL)
