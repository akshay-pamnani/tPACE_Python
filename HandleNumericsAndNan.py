
import numpy as np

def handle_numerics_and_nan(Ly, Lt):
    # Check for the presence of NA and remove them (if they exist) from the two lists in a pairwise manner
    if np.any(np.isnan(np.concatenate(Lt))) or np.any(np.isnan(np.concatenate(Ly))):
        def helper(x):
            return np.where(~np.isnan(x))[0]

        L = [(Ly[i], Lt[i]) for i in range(len(Ly))]
        valid_indexes = [np.intersect1d(helper(x[0]), helper(x[1])) for x in L]

        Ly = [np.array(Ly[i])[valid_indexes[i]] for i in range(len(Ly))]
        Lt = [np.array(Lt[i])[valid_indexes[i]] for i in range(len(Ly))]

        if any([len(x) == 0 for x in Ly]):
            raise ValueError('Subjects with only NA values are not allowed.')

        ni_y = [sum(~np.isnan(np.array(x))) for x in Ly]
        if all(ni == 1 for ni in ni_y):
            raise ValueError("FPCA is aborted because the data do not contain repeated measurements in y!")

    Ly = [np.array(x, dtype=float) for x in Ly]
    Lt = [np.array(x, dtype=float) for x in Lt]
    Lt = [np.around(x, decimals=14) for x in Lt]

    return {'Ly': Ly, 'Lt': Lt}
