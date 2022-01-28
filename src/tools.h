#ifndef TOOLS_H
#define TOOLS_H
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]


/** --------------------- calculation functions ---------------------------- **/
double lse(arma::vec arr, int count);
double lse(std::vector<double> arr, int count);

/** ---------------------- reshaping functions ----------------------------- **/
arma::mat vec2mat(arma::vec u, Rcpp::List& params); 

#endif
