from isRegular import isRegular
from scipy.linalg import issymmetric


def check_options(t, optns, n):
    if optns.get('useBinnedData') not in ['FORCE', 'AUTO', 'OFF']:
        raise ValueError("FPCA is aborted because the argument: useBinnedData is invalid!")

    if not (isinstance(optns.get('userBwMu'), (int, float)) and optns.get('userBwMu') >= 0):
        raise ValueError("FPCA is aborted because the argument: userBwMu is invalid!")

    if optns.get('methodBwMu') not in ['Default', 'CV', 'GCV', 'GMeanAndGCV']:
        raise ValueError("FPCA is aborted because the argument: methodBwMu is invalid!")

    if not (isinstance(optns.get('userBwCov'), (int, float)) and optns.get('userBwCov') >= 0):
        raise ValueError("FPCA is aborted because the argument: userBwCov is invalid!")

    if optns.get('methodBwCov') not in ['Default', 'CV', 'GCV', 'GMeanAndGCV']:
        raise ValueError("FPCA is aborted because the argument: methodBwCov is invalid!")

    if 'kFoldMuCov' in optns and (not isinstance(optns['kFoldMuCov'], (int, float)) or optns['kFoldMuCov'] < 2):
        raise ValueError("Invalid `kFoldMuCov` option")

    if isinstance(optns.get('methodSelectK'), str):
        if optns['methodSelectK'] not in ['FVE', 'AIC', 'BIC']:
            raise ValueError("FPCA is aborted because the argument: methodSelectK is invalid!")
    elif isinstance(optns.get('methodSelectK'), (int, float)):
        if int(optns['methodSelectK']) != optns['methodSelectK'] or optns['methodSelectK'] <= 0 or optns['methodSelectK'] > n:
            raise ValueError("FPCA is aborted because the argument: methodSelectK is invalid!")
    else:
        raise ValueError("FPCA is aborted because the argument: methodSelectK is invalid!")

    if 'FVEthreshold' in optns:
        if not (0 <= optns['FVEthreshold'] <= 1):
            raise ValueError("FPCA is aborted because the argument: FVEthreshold is invalid!")

    if 'FVEfittedCov' in optns:
        if not (0 <= optns['FVEfittedCov'] <= 1):
            raise ValueError("FPCA is aborted because the argument: FVEfittedCov is invalid!")

    if not (isinstance(optns.get('maxK'), (int, float)) and 1 <= optns['maxK'] <= n):
        raise ValueError("FPCA is aborted because the argument: maxK is invalid!")

    if optns.get('dataType') not in [None, "Sparse", "DenseWithMV", "Dense", "p>>n"]:
        raise ValueError("FPCA is aborted because the argument: dataType is invalid!")

    if optns.get('dataType') is None:
        optns['dataType'] = isRegular(t)

    if not isinstance(optns.get('error'), bool):
        raise ValueError("FPCA is aborted because the error option is invalid!")

    if not (isinstance(optns.get('nRegGrid'), (int, float)) and optns['nRegGrid'] > optns['maxK']):
        raise ValueError("FPCA is aborted because the argument: nRegGrid is invalid!")

    if optns.get('methodXi') not in ['CE', 'IN']:
        raise ValueError("FPCA is aborted because the argument: methodXi is invalid!")

    if optns.get('kernel') not in ['epan', 'gauss', 'rect', 'quar', 'gausvar']:
        raise ValueError("FPCA is aborted because the argument: kernel is invalid!")

    if optns.get('numBins') is not None and (not isinstance(optns['numBins'], (int, float)) or optns['numBins'] <= 1):
        raise ValueError("FPCA is aborted because the argument: numBins is invalid!")

    if optns.get('useBinnedData') == 'FORCE' and optns.get('numBins') is None:
        raise ValueError("FPCA is aborted because the argument: numBins is NULL but you FORCE binning!")

    if not isinstance(optns.get('yname'), str):
        raise ValueError("FPCA is aborted because the argument: yname is invalid!")

    if not isinstance(optns.get('plot'), bool):
        raise ValueError("FPCA is aborted because the argument: plot is invalid!")

    if optns.get('methodRho') not in ['trunc', 'ridge', 'vanilla']:
        raise ValueError("FPCA is aborted because the argument: methodRho is invalid!")

    if not isinstance(optns.get('verbose'), bool):
        raise ValueError("FPCA is aborted because the argument: verbose is invalid!")

    if 'userMu' in optns and optns['userMu'] is not None:
        user_mu = optns['userMu']
        if not (isinstance(user_mu, dict) and isinstance(user_mu.get('t'), list) and isinstance(user_mu.get('mu'), list) and len(user_mu['t']) == len(user_mu['mu'])):
            raise ValueError("FPCA is aborted because the argument: userMu is invalid!")

    if 'userCov' in optns and optns['userCov'] is not None:
        user_cov = optns['userCov']
        if not (isinstance(user_cov, dict) and isinstance(user_cov.get('t'), list) and isinstance(user_cov.get('cov'), list) and len(user_cov['t']) == len(user_cov['cov'][0]) and issymmetric(user_cov['cov'])):
            raise ValueError("FPCA is aborted because the argument: userCov is invalid! (eg. Check if 'cov' is symmetric and 't' is of appropriate size.)")

    if 'userSigma2' in optns:
        if not (isinstance(optns['userSigma2'], (int, float)) and optns['userSigma2'] >= 0):
            raise ValueError("userSigma2 invalid.")
        if optns['userSigma2'] == 0 and optns['error']:
            raise ValueError("userSigma2 specified to be 0 but error = TRUE. If no measurement error is assumed then use error = FALSE.")

    if 'userRho' in optns:
        if not (isinstance(optns['userRho'], (int, float)) and optns['userRho'] >= 0):
            raise ValueError("userSigma2 invalid.")
        if optns['userRho'] == 0 and optns['error']:
            raise ValueError("userRho specified to be 0 but error = TRUE. If no measurement error is assumed then use error = FALSE.")

    if not (isinstance(optns.get('outPercent'), list) and len(optns['outPercent']) == 2 and all(0 <= x <= 1 for x in optns['outPercent'])):
        raise ValueError("FPCA is aborted because the argument: outPercent is invalid!")

    if not (isinstance(optns.get('rotationCut'), list) and len(optns['rotationCut']) == 2 and all(0 <= x <= 1 for x in optns['rotationCut'])):
        raise ValueError("FPCA is aborted because the argument: rotationCut is invalid!")

    if 'userCov' in optns and isinstance(optns['userCov'], bool):
        raise ValueError("FPCA is aborted because the argument: userCov is invalid!")

    if optns.get('methodMuCovEst') not in ['smooth', 'cross-sectional']:
        raise ValueError("FPCA is aborted because the argument: methodMuCovEst is invalid!")

    if not isinstance(optns.get('lean'), bool):
        raise ValueError("FPCA is aborted because the lean option is invalid!")

    if not isinstance(optns.get('useBW1SE'), bool):
        raise ValueError("FPCA is aborted because the useBW1SE option is invalid!")

    return True

