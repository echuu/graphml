
# algo_helpers.R


## functions in this file
## TODO: add function documentation/description
##
##     preprocess()
##     l1_norm()
##     l2_norm()
##     log_det()
##     log_sum_exp()
##     extractSupport()
##     log_int_rect()
##     log_int_rect()
##     logQ()
##     logJ()
##     logMSE()
##





preprocess = function(post_samps, D, params = NULL) {

    psi_u = apply(post_samps, 1, psi, params = params) %>% unname() # (J x 1)

    # (1.2) name columns so that values can be extracted by partition.R
    u_df_names = character(D + 1)
    for (d in 1:D) {
        u_df_names[d] = paste("u", d, sep = '')
    }
    u_df_names[D + 1] = "psi_u"

    # populate u_df
    u_df = cbind(post_samps, psi_u) # J x (D + 1)
    names(u_df) = u_df_names

    return(u_df)
} # end of preprocess() function -----------------------------------------------





l1_norm = function(u, u_0) {
    sum(abs(u - u_0))
}





l2_norm = function(u, u_0) {
    sum((u - u_0)^2)
}





#### log_det() -----------------------------------------------------------------
#
#
log_det = function(xmat) {
    return(c(determinant(xmat, logarithm = T)$modulus))
}
# end log_det() function -------------------------------------------------------





#### log_sum_exp() () ----------------------------------------------------------
# calculates expressions of the form log(sum(exp(x)))
#
log_sum_exp = function(x) {
    offset = max(x)                         # scale by max to prevent overflow
    s = log(sum(exp(x - offset))) + offset
    i = which(!is.finite(s))                # check for any overflow
    if (length(i) > 0) {                    # replace inf values with max
        s[i] = offset
    }

    return(s)
}
# end of log_sum_exp() function ------------------------------------------------





#### extractSupport() ----------------------------------------------------------
#
#
extractSupport = function(u_df, D) {

    # (3.1) obtain the (data-defined) support for each of the parameters
    param_support = matrix(NA, D, 2) # store the parameter supports row-wise

    for (d in 1:D) {
        param_d_min = min(u_df[,d])
        param_d_max = max(u_df[,d])

        param_support[d,] = c(param_d_min, param_d_max)
    }

    return(param_support)
}
# end of extractSupport() function ---------------------------------------------





#### log_int_rect() ------------------------------------------------------------
# compute the log of the closed form integral over the d-th rectangle
# note: we don't compute the integral explicitly b/c we use log-sum-exp at the
# end of the calculation for stability
log_int_rect = function(l_d, a, b) {


    if (l_d > 0) {
        # extract e^(-lambda_d * a), term corresponding to the lower bound
        out = - l_d * a - log(l_d) + VGAM::log1mexp(l_d * b - l_d * a)
    } else {
        # extract e^(-lambda_d * b), term corresponding to the upper bound
        out = - l_d * b - log(-l_d) + VGAM::log1mexp(l_d * a - l_d * b)
    }
    return(out)
}
# end log_int_rect() function --------------------------------------------------





## logQ() function -------------------------------------------------------------
# c_star : value whose cost function is to be evaluated
# c_k    : L-dim vector of function evaluations in the k-th partition
logQ = function(c_star, c_k) {

    L = length(c_k)

    log_rel_error = rep(NA, L) # store the log relative error

    # for (l in 1:L) {
    #     ## perform a stable calculation of log(abs(1-exp(-c_star + c_k[l])))
    #     ## by considering cases when c_star > c_k[l], c_star < c_k[l]
    #     if (c_star > c_k[l])
    #         log_rel_error[l] = log1mexp(c_star - c_k[l])
    #     else if (c_star < c_k[l])
    #         log_rel_error[l] = log1mexp(c_k[l] - c_star) - c_star + c_k[l]
    #
    #     # if c_star == c_k[l_k] : do nothing; NA value will be skipped over in
    #     # final calculation
    #
    # } # end of for loop iterating over each element in k-th partition

    sign_ind = (c_star < c_k)
    log_rel_error = VGAM::log1mexp(abs(c_star - c_k)) + (c_k - c_star) * sign_ind

    return(log_sum_exp(log_rel_error[!is.na(log_rel_error)]))

}
# end of logQ() function -------------------------------------------------------





logJ = function(c_star, c_k) {

    L = length(c_k)

    log_rel_error = rep(NA, L) # store the log relative error

    # for (l in 1:L) {
    #     ## perform a stable calculation of log(abs(1-exp(-c_star + c_k[l])))
    #     ## by considering cases when c_star > c_k[l], c_star < c_k[l]
    #     if (c_star < c_k[l])
    #         log_rel_error[l] = log1mexp(c_k[l] - c_star)
    #     else if (c_star > c_k[l])
    #         log_rel_error[l] = log1mexp(c_star - c_k[l]) + c_star - c_k[l]
    #
    #     # if c_star == c_k[l_k] : do nothing; NA value will be skipped over in
    #     # final calculation
    #
    # } # end of for loop iterating over each element in k-th partition

    sign_ind = (c_star > c_k)
    log_rel_error = VGAM::log1mexp(abs(c_star - c_k)) + (c_star - c_k) * sign_ind

    return(log_sum_exp(log_rel_error[!is.na(log_rel_error)]))

}




## logMSE() function -------------------------------------------------------------
# c_star : value whose cost function is to be evaluated
# c_k    : L-dim vector of function evaluations in the k-th partition
logMSE = function(c_star, c_k) {

    L = length(c_k)

    log_mse = rep(NA, L) # store the log mse

    for (l in 1:L) {
        if (c_star > c_k[l]) {
            log_mse[l] = 2 * (-c_k[l] + VGAM::log1mexp(c_star - c_k[l]))
        } else if (c_star < c_k[l]) {
            log_mse[l] = 2 * (-c_star + VGAM::log1mexp(c_k[l] - c_star))
        }
    }

    return(- log(L) + log_sum_exp(log_mse[!is.na(log_mse)]))

}
# end logMSE() function --------------------------------------------------------
# end of algo_helpers.R





# end of algo_helpers.R
