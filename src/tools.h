#ifndef TOOLS_H
#define TOOLS_H
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

Rcpp::List fitTree(Rcpp::DataFrame x, Rcpp::Formula formula);
Rcpp::List getPartition(Rcpp::List tree, arma::mat supp);

#endif
