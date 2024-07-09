import numpy as np

def isRegular(t):
    # Check the data type in terms of dense-sparse. Classification is dense (2), or  data with missing values (1) or sparse (0) data
    # t : n-by-1 list of vectors 

    tt = []
    for item in t:
        if isinstance(item, (list, np.ndarray)):
            tt.extend(item)
        else:
            tt.append(item)

    f = len(tt) / len(set(tt)) / len(t)
    
    if f == 1:
        if len(set(tt)) < 8:  # In case of low number of observations per subject
            return 'Sparse'
        else:
            return 'Dense'  # For either regular and irregular data
    elif f > 0.80:
        return 'DenseWithMV'
    else:
        return 'Sparse'