import numpy as np

def get_bin_num(n, m, data_type, verbose):
    """
    Get the number of bins
    
    Parameters:
    n : int
        Number of curves
    m : int or float
        Median or max value of number of time-points
    data_type : str
        Indicator about structure of the data (dense (2), or dataType data with missing values (1) or sparse (0))
    verbose : bool
        Output diagnostics/progress
    
    Returns:
    int or None : Number of bins or None if no binning is performed
    """
    num_bin = None
    
    if m <= 20:
        if data_type == 'Sparse':
            string = 'Median of ni'
        else:
            string = 'Maximum of ni'
        if verbose:
            print(f"{string} is no more than 20! No binning is performed!")
        return None

    if m > 400:
        num_bin = 400

    if n > 5000:
        mstar = max(20, (((5000 - n) * 19) / 2250) + 400)
        if mstar < m:
            num_bin = int(np.ceil(mstar))
        else:
            if verbose:
                print('No binning is needed!')
            return None

    if verbose and num_bin is None:
        print('No binning is needed!')
    
    return num_bin
