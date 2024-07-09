import numpy as np
from collections import defaultdict
import warnings
from GetBinNum import get_bin_num
from GetBinnedCurve import get_binned_curve

def get_binned_dataset(y, t, optns):
    # Initialize output dictionary
    bin_data_output = {'newy': None, 'newt': None}
    t = [ti if isinstance(ti, (list, np.ndarray)) else np.atleast_1d(ti) for ti in t]
    y = [yi if isinstance(yi, (list, np.ndarray)) else np.atleast_1d(yi) for yi in y]
    
    # Extract options
    data_type = optns['dataType']
    verbose = optns['verbose']
    num_bins = optns['numBins']
    use_binned_data = optns['useBinnedData']
    
    # Flatten t into a single list
    tt = np.concatenate(t)
    a0 = np.min(tt)
    b0 = np.max(tt)
    n = len(t)
    ni = [len(ti) if isinstance(ti, (list, np.ndarray)) else 1 for ti in t]
    
    if data_type == 'Sparse':
        m = np.median(ni)
    else:
        m = np.max(ni)
    
    # Determine number of bins automatically if numBins is None and useBinnedData is 'AUTO'
    if num_bins is None and use_binned_data == 'AUTO':
        num_bins = get_bin_num(n, m, data_type, verbose)
        if num_bins is None:
            bin_data_output['newt'] = t
            bin_data_output['newy'] = y
            return bin_data_output
        elif use_binned_data == 'AUTO':
            warnings.warn('Automatically binning measurements. To turn off this warning set option useBinnedData to \'FORCE\' or \'OFF\'')
    
    # Otherwise, use the provided number of bins (ceiling)
    num_bins = np.ceil(num_bins).astype(int)
    
    # Perform binning for each dataset
    res_list = [get_binned_curve(ti, yi, num_bins, True, True, [a0, b0]) for ti, yi in zip(t, y)]
    bin_data_output['newt'] = [res['midpoint'] for res in res_list]
    bin_data_output['newy'] = [res['newy'] for res in res_list]
    
    return bin_data_output


