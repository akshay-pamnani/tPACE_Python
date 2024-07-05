def set_options(y, t, optns):
    # Extract options with defaults
    method_mu_cov_est = optns.get('methodMuCovEst')
    user_bw_mu = optns.get('userBwMu')
    method_bw_mu = optns.get('methodBwMu')
    user_bw_cov = optns.get('userBwCov')
    method_bw_cov = optns.get('methodBwCov')
    k_fold_mu_cov = optns.get('kFoldMuCov')
    method_select_k = optns.get('methodSelectK')
    fve_threshold = optns.get('FVEthreshold')
    fve_fitted_cov = optns.get('FVEfittedCov')
    fit_eigen_values = optns.get('fitEigenValues')
    max_k = optns.get('maxK')
    data_type = optns.get('dataType')
    error = optns.get('error')
    n_reg_grid = optns.get('nRegGrid')
    method_xi = optns.get('methodXi')
    shrink = optns.get('shrink')
    kernel = optns.get('kernel')
    num_bins = optns.get('numBins')
    yname = optns.get('yname')
    method_rho = optns.get('methodRho')
    user_grid = optns.get('usergrid')
    user_rho = optns.get('userRho')
    diagnostics_plot = optns.get('diagnosticsPlot')
    plot = optns.get('plot', diagnostics_plot)
    if diagnostics_plot is not None:
        print("Warning: The option 'diagnosticsPlot' is deprecated. Use 'plot' instead")
    verbose = optns.get('verbose')
    user_mu = optns.get('userMu')
    out_percent = optns.get('outPercent')
    user_cov = optns.get('userCov')
    user_sigma2 = optns.get('userSigma2')
    rotation_cut = optns.get('rotationCut')
    use_binned_data = optns.get('useBinnedData')
    use_binned_cov = optns.get('useBinnedCov')
    lean = optns.get('lean')
    use_bw_1se = optns.get('useBW1SE')
    impute_scores = optns.get('imputeScores')

    if method_bw_mu is None:
        method_bw_mu = 'Default'
    if user_bw_mu is None:
        if method_bw_mu == 'Default':
            user_bw_mu = 0.05 * (max(t) - min(t))
        else:
            user_bw_mu = 0.0
    if method_bw_cov is None:
        method_bw_cov = 'Default'
    if user_bw_cov is None:
        if method_bw_cov == 'Default':
            user_bw_cov = 0.10 * (max(t) - min(t))
        else:
            user_bw_cov = 0.0
    if k_fold_mu_cov is None:
        k_fold_mu_cov = 10
    if method_select_k is None:
        method_select_k = "FVE"
    if fve_threshold is None:
        fve_threshold = 0.99
    if fve_fitted_cov is None:
        fve_fitted_cov = None
    if data_type is None:
        data_type = is_regular(t)
    if fit_eigen_values is None:
        fit_eigen_values = False
    if method_mu_cov_est is None:
        if data_type == 'Sparse':
            method_mu_cov_est = 'smooth'
        else:
            method_mu_cov_est = 'cross-sectional'
    if fit_eigen_values and data_type == 'Dense':
        raise ValueError('Fit method only applies to sparse data')
    if error is None:
        error = True
    if n_reg_grid is None:
        if data_type in ['Dense', 'DenseWithMV']:
            n_reg_grid = len(set(round(x, 6) for x in t))
        else:
            n_reg_grid = 51
    if max_k is None:
        max_k = min(n_reg_grid - 2, len(y) - 2)
        if method_mu_cov_est == 'smooth':
            max_k = min(max_k, 20)
        if max_k < 1:
            print("Automatically defined maxK cannot be less than 1. Reset to maxK = 1 now!")
            max_k = 1
        if len(y) <= 3:
            print("The sample size is less or equal to 3 curves. Be cautious!")
    if method_xi is None:
        if data_type == 'Dense':
            method_xi = "IN"
        elif data_type == 'Sparse':
            if min(len(ti) for ti in t) > 20:
                tt = [item for sublist in t for item in sublist]
                t_min = min(tt)
                t_max = max(tt)
                spacing_max = max(max(ti[1] - t_min, max(ti[1:]) - ti[0], t_max - ti[-1]) for ti in t)
                if spacing_max <= 0.06 * (max(tt) - min(tt)):
                    method_xi = "IN"
                else:
                    method_xi = "CE"
            else:
                method_xi = "CE"
        elif data_type == 'DenseWithMV':
            method_xi = "CE"
        else:
            method_xi = "IN"
    if shrink is None:
        shrink = False
    if shrink and (error is not True or method_xi != "IN"):
        print('shrinkage method only has effects when methodXi = "IN" and error = TRUE! Reset to shrink = FALSE now!')
        shrink = False
    if kernel is None:
        if data_type == "Dense":
            kernel = "epan"
        else:
            kernel = "gauss"
    kern_names = ["rect", "gauss", "epan", "gausvar", "quar"]
    if kernel not in kern_names:
        print(f'kernel {kernel} is unrecognizable! Reset to automatic selection now!')
        kernel = None
    if kernel is None:
        if data_type in ["Dense", "DenseWithMV"]:
            kernel = "epan"
        else:
            kernel = "gauss"
    if yname is None:
        yname = 'y'
    if max_k > (n_reg_grid - 2):
        print(f"maxK can only be less than or equal to {n_reg_grid-2}! Reset to be {n_reg_grid-2} now!")
        max_k = n_reg_grid - 2
    if isinstance(method_select_k, int):
        fve_threshold = 1
        if method_select_k > (n_reg_grid - 2):
            print(f"maxK can only be less than or equal to {n_reg_grid-2}! Reset to be {n_reg_grid-2} now!")
            max_k = n_reg_grid - 2
        elif method_select_k <= 0:
            print("methodSelectK must be a positive integer! Reset to BIC now!")
            method_select_k = "BIC"
            fve_threshold = 0.95
    if plot is None:
        plot = False
    if method_rho is None:
        method_rho = 'vanilla'
    if user_rho is None:
        user_rho = None
    if verbose is None:
        verbose = False
    if user_mu is None:
        user_mu = None
    if user_cov is None:
        user_cov = None
    if out_percent is None:
        out_percent = [0, 1]
    if rotation_cut is None:
        rotation_cut = [0.25, 0.75]
    if num_bins is not None and num_bins < 10:
        print("Number of bins must be at least 10!")
        num_bins = None
    if use_binned_data is None:
        use_binned_data = 'AUTO'
    if use_binned_cov is None:
        use_binned_cov = True
        if (128 > len(y)) and (3 > sum(len(yi) for yi in y) / len(y)):
            use_binned_cov = False
    if user_grid is None:
        user_grid = False
    if lean is None:
        lean = False
    if use_bw_1se is None:
        use_bw_1se = False
    if impute_scores is None:
        impute_scores = True

    ret_optns = {
        'userBwMu': user_bw_mu, 'methodBwMu': method_bw_mu, 'userBwCov': user_bw_cov, 'methodBwCov': method_bw_cov,
        'kFoldMuCov': k_fold_mu_cov, 'methodSelectK': method_select_k, 'FVEthreshold': fve_threshold, 'FVEfittedCov': fve_fitted_cov,
        'fitEigenValues': fit_eigen_values, 'maxK': max_k, 'dataType': data_type, 'error': error, 'shrink': shrink,
        'nRegGrid': n_reg_grid, 'rotationCut': rotation_cut, 'methodXi': method_xi, 'kernel': kernel,
        'lean': lean, 'diagnosticsPlot': diagnostics_plot, 'plot': plot, 'numBins': num_bins, 'useBinnedCov': use_binned_cov,
        'usergrid': user_grid, 'yname': yname, 'methodRho': method_rho, 'verbose': verbose, 'userMu': user_mu, 'userCov': user_cov}
