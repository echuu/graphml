#ifndef GWISH_H
#define GWISH_H
#include <RcppArmadillo.h>
#include "graphml_types.h"
#include <Rcpp.h>
// [[Rcpp::depends(RcppArmadillo)]]


Rcpp::List init_graph(arma::umat G, u_int b, arma::mat V);

double generalApprox(arma::umat G, u_int b, arma::mat V, u_int J);

double approx_v1(Rcpp::DataFrame u_df,
				 arma::vec uStar,
				 arma::mat data,
				 Rcpp::List& params);

double approxZ(Rcpp::List& params,
	arma::vec leaf,
	std::unordered_map<int, arma::vec> candidates,
	std::unordered_map<int, arma::vec> bounds,
	u_int K);

double approx_integral(u_int K, arma::mat& psi_df, arma::mat& bounds,
	Rcpp::List& params);

#endif
