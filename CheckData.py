

#' Check data format
#' 
#' Check if there are problems with the form and basic structure of the functional data 'y' and the recorded times 't'.
#' 
#' @param y is a n-by-1 list of  vectors
#' @param t is a n-by-1 list of vectors
#' @export

import numpy as np

def CheckData(y, t):
    
    if not isinstance(y, list):
        raise ValueError('y should be a list')
    if not isinstance(t, list):
        raise ValueError('t should be a list')
    
    if len(t) != len(y):
        raise ValueError('t and y should have the same length')
    
    if not all(isinstance(x, (int, float)) for lst in y for x in lst):
        raise ValueError("FPCA is aborted because 'y' members are not all of type double or integer!")
    
    if not all(isinstance(x, (int, float)) for lst in t for x in lst):
        raise ValueError("FPCA is aborted because 't' members are not all of type double or integer!")
    
    ni_y = [sum(~np.isnan(np.array(x))) for x in y]
    if all(ni == 1 for ni in ni_y):
        raise ValueError("FPCA is aborted because the data do not contain repeated measurements in y!")
    
    ni_tt = [sum(~np.isnan(np.array(x))) for x in t]
    if all(ni == 1 for ni in ni_tt):
        raise ValueError("FPCA is aborted because the data do not contain repeated measurements in t!")
        
    if any(len(lst) != len(set(lst)) for lst in t):
        raise ValueError("FPCA is aborted because within-subject 't' members have duplicated values.")
    
    if not all([all(np.diff(lst) >= 0) for lst in t]):
        raise ValueError('Each vector in t should be in ascending order')

    if np.min(np.concatenate(y)) == -np.inf:
        raise ValueError('There are entries in y which are -Inf')
    
    if np.max(np.concatenate(y)) == np.inf:
        raise ValueError('There are entries in y which are Inf')
    
    if np.max(np.diff(sorted(np.concatenate(t)))) / (np.max(np.concatenate(t)) - np.min(np.concatenate(t))) > 0.1:
        print('Warning: There is a time gap of at least 10% of the observed range across subjects')


