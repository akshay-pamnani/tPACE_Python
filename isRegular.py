def isRegular(t):
    # Check the data type in terms of dense-sparse. Classification is dense (2), or  data with missing values (1) or sparse (0) data
    # t : n-by-1 list of vectors 

    tt = [item for sublist in t for item in sublist]
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