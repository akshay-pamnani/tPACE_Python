import numpy as np
import pandas as pd
from collections import namedtuple

def RCPPmean(values):
    return np.mean(values)

def RCPPvar(values):
    return np.var(values, ddof=1)  # ddof=1 for sample variance

def BinRawCov(rcov):
    # Ignoring the class condition
    # if 'RawCC' in rcov['class']:
    #     rcov['cxxn'] = rcov['rawCCov']
    #     rcov['tPairs'] = rcov['tpairn']

    # Aggregate counts, mean raw covariances, and RSS
    tmp = (
        pd.DataFrame({
            'time1': rcov['tPairs'][:, 0],
            'time2': rcov['tPairs'][:, 1],
            'values': rcov['cxxn']
        })
        .groupby(['time1', 'time2'])
        .agg(
            mean_val=('values', lambda yy: RCPPmean(yy)),
            count=('values', 'size'),
            RSS=('values', lambda yy: RCPPvar(yy) * (len(yy) - 1))
        )
        .reset_index()
    )

    tPairs = tmp[['time1', 'time2']].values
    summaryDat = tmp[['mean_val', 'count', 'RSS']].values
    meanVals = summaryDat[:, 0]
    count = summaryDat[:, 1]
    RSS = summaryDat[:, 2]
    RSS[np.isnan(RSS)] = 0

    diagRSS = diagCount = diagMeans = tDiag = None
    if rcov.get('diag') is not None:
        diag_tmp = (
            pd.DataFrame({
                'time': rcov['diag'][:, 0],
                'values': rcov['diag'][:, 1]
            })
            .groupby('time')
            .agg(
                mean_val=('values', lambda yy: RCPPmean(yy)),
                count=('values', 'size'),
                RSS=('values', lambda yy: RCPPvar(yy) * (len(yy) - 1))
            )
            .reset_index()
        )

        tDiag = diag_tmp['time'].values
        diagSummary = diag_tmp[['mean_val', 'count', 'RSS']].values
        diagMeans = diagSummary[:, 0]
        diagCount = diagSummary[:, 1]
        diagRSS = diagSummary[:, 2]
        diagRSS[np.isnan(diagRSS)] = 0

    # Define the result structure
    BinnedRawCov = namedtuple(
        'BinnedRawCov', 
        ['tPairs', 'meanVals', 'RSS', 'tDiag', 'diagMeans', 'diagRSS', 'count', 'diagCount', 'error', 'dataType']
    )
    print("Aggregated tPairs:\n", tPairs)
    print("Aggregated DataFrame (tmp):\n", tmp)


    result = BinnedRawCov(
        tPairs=tPairs,
        meanVals=meanVals,
        RSS=RSS,
        tDiag=tDiag,
        diagMeans=diagMeans,
        diagRSS=diagRSS,
        count=count,
        diagCount=diagCount,
        error=rcov.get('error'),
        dataType=rcov.get('dataType')
        # Ignoring the class condition
        # class_name=rcov.get('class')
    )

    # Ignoring the class condition
    # if 'RawCC' in rcov['class']:
    #     result = result._replace(class_name='BinnedRawCC')
    # else:
    #     result = result._replace(class_name='BinnedRawCov')

    return result
