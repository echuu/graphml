#ifndef RGWISHART_H
#define RGWISHART_H
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

arma::mat rwish_c(arma::mat Ts, unsigned int b, unsigned int p);
arma::mat rgwish_c(arma::mat G, arma::mat Ts, unsigned int b, unsigned int p, 
	double threshold_c);
arma::mat rgw(unsigned int J, Rcpp::List& obj);

#endif