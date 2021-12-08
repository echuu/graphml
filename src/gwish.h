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

double approxWrapper(arma::mat data, arma::vec locs, arma::vec uStar, u_int D,
	arma::mat bounds, arma::vec leafId, Rcpp::List& params);


/** utility functions **/
arma::vec chol2vec(arma::mat& M, u_int D);
arma::mat create_psi_mat_cpp(arma::vec u, Rcpp::List& params);
double xi(u_int i, u_int j, arma::mat& L);

/** ------ objective function evaluation ------- **/
double psi_cpp_mat(arma::mat& psi_mat, Rcpp::List& params);
double psi_cpp(arma::vec& u, Rcpp::List& params);

arma::vec calcMode(arma::mat u_df, Rcpp::List& params);

/** ------ updated grad/hess functions for non-diagonal scale matrix ------- **/
arma::vec grad_gwish(arma::mat& psi, Rcpp::List& params);
double dpsi_ij(u_int i, u_int j, arma::mat& psi_mat, Rcpp::List& params);
double dpsi(u_int r, u_int s, u_int i, u_int j,
	arma::mat& psi, Rcpp::List& params);

arma::mat hess_gwish(arma::mat& psi_mat, Rcpp::List& params);
double d2psi_ijkl(u_int i, u_int j, u_int k, u_int l,
	arma::mat& psi, Rcpp::List& params);
double d2psi(u_int r, u_int s, u_int i, u_int j, u_int k, u_int l,
	arma::mat& psi, Rcpp::List& params);




#endif
