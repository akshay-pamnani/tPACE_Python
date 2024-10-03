def check_svd_options(Ly1, Lt1, Ly2, Lt2, svd_opts):
    """
    Check the validity of SVD options for functional data analysis.

    Parameters:
    - Ly1: Placeholder for the first data matrix (not used in the function).
    - Lt1: Placeholder for the first time points (not used in the function).
    - Ly2: Placeholder for the second data matrix (not used in the function).
    - Lt2: Placeholder for the second time points (not used in the function).
    - svd_opts: A dictionary containing the SVD options with the following keys:
      - 'dataType1': Data type for the first dataset.
      - 'userMu1': User specified mean function for the first dataset.
      - 'dataType2': Data type for the second dataset.
      - 'userMu2': User specified mean function for the second dataset.
      - 'methodSelectK': Method for selecting K.
      - 'regulRS': Regularization option.

    Raises:
    - ValueError: If any of the conditions are not met.
    """

    if ((svd_opts.get('dataType1') == 'Sparse' and svd_opts.get('userMu1') is None) or
        (svd_opts.get('dataType2') == 'Sparse' and svd_opts.get('userMu2') is None)):
        raise ValueError('User specified mean function required for sparse functional data for cross covariance estimation.')

    method_select_k = svd_opts.get('methodSelectK')
    if isinstance(method_select_k, (int, float)):
        if method_select_k != round(method_select_k) or method_select_k <= 0:
            raise ValueError("FSVD is aborted: 'methodSelectK' is invalid!")

    regul_rs = svd_opts.get('regulRS')
    if regul_rs not in ['sigma2', 'rho']:
        raise ValueError("FSVD is aborted: Unknown regularization option. The argument 'regulRS' should be 'rho' or 'sigma2'!")
