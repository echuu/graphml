#ifndef EPMGP_H
#define EPMGP_H

#include "RcppArmadillo.h"
#define RCPP_ARMADILLO_RETURN_COLVEC_AS_VECTOR
#include "TruncParams.h"
// [[Rcpp::depends(RcppArmadillo)]]

// struct TruncParams;

double erfcx (double x);
Rcpp::List trunc_norm_moments(arma::vec lb_in, arma::vec ub_in,
                              arma::vec mu_in, arma::vec sigma_in);

TruncParams* tn(arma::vec lb_in, arma::vec ub_in,
				arma::vec mu_in, arma::vec sigma_in);

double ep_logz(arma::vec m, arma::mat K, arma::vec lb, arma::vec ub);

double ep(arma::vec m, arma::mat K, arma::vec lb, arma::vec ub);
#endif
