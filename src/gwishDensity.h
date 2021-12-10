#ifndef GWISHDENSITY_H
#define GWISHDENSITY_H
#include "graphml_types.h"
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

/** ----------- gwish density, gradient, hessian, root finding  ------------ **/



/** ---------------------  gwish density functions ------------------------- **/



/** ---------------------  gradient functions ------------------------------ **/

arma::vec grad_gwish(arma::mat& psi, Rcpp::List& params);
double dpsi_ij(u_int i, u_int j, arma::mat& psi_mat, Rcpp::List& params);
double dpsi(u_int r, u_int s, u_int i, u_int j,
	arma::mat& psi, Rcpp::List& params);

/** ----------------------  hessian functions ------------------------------ **/

arma::mat hess_gwish(arma::mat& psi_mat, Rcpp::List& params);
double d2psi_ijkl(u_int i, u_int j, u_int k, u_int l,
	arma::mat& psi, Rcpp::List& params);
double d2psi(u_int r, u_int s, u_int i, u_int j, u_int k, u_int l,
	arma::mat& psi, Rcpp::List& params);


/** -----------------  newton's method for root-finding  ------------------- **/






#endif
