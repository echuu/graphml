#ifndef GWISH_H
#define GWISH_H
#include <RcppArmadillo.h>
#include "graphml_types.h"
#include <Rcpp.h>
// [[Rcpp::depends(RcppArmadillo)]]


double generalApprox(arma::umat G, u_int b, arma::mat V, u_int J);

double approxZ(Rcpp::List& params,
	arma::vec leaf,
	std::unordered_map<int, arma::vec> candidates,
	std::unordered_map<int, arma::vec> bounds,
	u_int K);


/** utility functions **/
arma::mat create_psi_mat_cpp(arma::vec u, Rcpp::List& params);

/** ------ objective function evaluation ------- **/
double psi_cpp_mat(arma::mat& psi_mat, Rcpp::List& params);
double psi_cpp(arma::vec& u, Rcpp::List& params);

arma::vec calcMode(arma::mat u_df, Rcpp::List& params);


#endif
