
#ifndef OGAPPROX_H
#define OGAPPROX_H


#include "graphml_types.h"

/* ----------------------- main 3 algorithm functions ----------------------- */
double generalApprox(arma::umat G, u_int b, arma::mat V, u_int J);
double approx_v1(Rcpp::DataFrame u_df, arma::vec uStar, arma::mat data,
				 Rcpp::List& params);

double approxZ(Rcpp::List& params, arma::vec leaf,
	std::unordered_map<u_int, arma::vec> candidates,
	std::unordered_map<u_int, arma::vec> bounds,
	u_int K);


/* ----------------------- algorithm helper functions ----------------------- */




#endif