#ifndef TOOLS_H
#define TOOLS_H
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

Rcpp::List fitTree(Rcpp::DataFrame x, Rcpp::Formula formula);
Rcpp::List getPartition(Rcpp::List tree, arma::mat supp);
Rcpp::DataFrame mat2df(arma::mat x);
Rcpp::StringVector createDfName(unsigned int D);
Rcpp::DataFrame mat2df(arma::mat x, Rcpp::StringVector nameVec);
double xi(u_int i, u_int j, arma::mat& L);

arma::vec matrix2vector(arma::mat m, const bool byrow=false);
arma::mat getFreeElem(arma::umat G, u_int p);
arma::mat getNonFreeElem(arma::umat G, u_int p, u_int n_nonfree);
double lse(arma::vec arr, int count);

#endif
