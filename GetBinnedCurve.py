import numpy as np

def get_binned_curve(x, y, M=10, isMnumBin=True, nonEmptyOnly=False, limits=None):
    # x : 1D array of values to be binned
    # y : 1D array of values corresponding to x
    # M : positive integer denotes the number of bins to be used or positive value of the width of each equidistant bin
    # isMnumBin : use number of bins (True) or bandwidth h (False)
    # nonEmptyOnly : output only non-empty bins (True)
    # limits : lower and upper limit of domain x (a0, b0)
    
    if limits is None:
        limits = [min(x), max(x)]
    
    if M < 0:
        raise ValueError("GetBinnedCurve is aborted because the argument: M is invalid!")
    
    if not isMnumBin:
        # Use bandwidth h
        h = M
        if h >= np.diff(np.ptp(x)):
            return get_res_m_is_one(x, y, h)
        
        xx = np.arange(limits[0], limits[1] + h, h)
        M = len(xx) - 1
        N = M + 1
        midpoint = xx[:N-1] + (h / 2)
        print(midpoint)
        newy, count = get_bins(x, y, xx)
        
        if nonEmptyOnly:
            mask = count > 0
            midpoint = midpoint[mask]
            newy = newy[mask]
            count = count[mask]
            M = len(midpoint)
        
        return {'midpoint': midpoint, 'newy': newy, 'count': count, 'M': M, 'h': h}
    
    else:
        # Use number of bins M
        M = int(np.ceil(M))
        if M == 1:
            return get_res_m_is_one(x, y)
        else:
            h = np.diff(limits) / M
            #xx = np.concatenate(([limits[0]], np.linspace(limits[0] + h / 2, limits[1] - h / 2, M - 1), [limits[1]]))
            xx = np.concatenate(([limits[0]], np.linspace(limits[0] + h / 2, limits[1] - h / 2, M - 1).reshape(-1), [limits[1]]))
            N = len(xx)
            midpoint = np.linspace(limits[0], limits[1], M)
            
            newy, count = get_bins(x, y, xx)
            
            if nonEmptyOnly:
                mask = count > 0
                midpoint = midpoint[mask]
                newy = newy[mask]
                count = count[mask]
                M = len(midpoint)
                
            return {'midpoint': midpoint, 'newy': newy, 'count': count, 'numBin': M, 'binWidth': h}

def get_bins(x, y, xx):
    N = len(xx)
    count = np.zeros(N-1)
    newy = np.full(N-1, np.nan)
    
    for i in range(1, N-1):
        ids = (x >= xx[i-1]) & (x < xx[i])
        if np.all(ids == 0):
            count[i-1] = 0
            newy[i-1] = np.nan
        else:
            count[i-1] = np.sum(ids)
            newy[i-1] = np.mean(y[ids])
    
    # For the last bin, include the left and right end point
    ids = (x >= xx[-2]) & (x <= xx[-1])
    if np.all(ids == 0):
        count[-1] = 0
        newy[-1] = np.nan
    else:
        count[-1] = np.sum(ids)
        newy[-1] = np.mean(y[ids])
    
    return newy, count

def get_res_m_is_one(x, y, h=None):
    if h is None:
        h = np.ptp(x)
    r = h
    M = 1
    midpoint = r * 0.5
    count = len(x)
    newy = np.mean(y)
    return {'midpoint': midpoint, 'newy': newy, 'count': count, 'M': M, 'h': h}
