#ifndef TOOLS_H
#define TOOLS_H
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]


/** ---------------------- R dataframe functions --------------------------- **/
Rcpp::StringVector createDfName(unsigned int D);
Rcpp::DataFrame mat2df(arma::mat x, Rcpp::StringVector nameVec);

arma::vec matrix2vector(arma::mat m, const bool byrow=false);

/** -------------------- gwish-specific functions -------------------------- **/
arma::mat getFreeElem(arma::umat G, u_int p);
arma::mat getNonFreeElem(arma::umat G, u_int p, u_int n_nonfree);

/** --------------------- calculation functions ---------------------------- **/
double xi(u_int i, u_int j, arma::mat& L);
double lse(arma::vec arr, int count);
double lse(std::vector<double> arr, int count);

/** ---------------------- reshaping functions ----------------------------- **/
arma::mat vec2mat(arma::vec u, Rcpp::List& params); 

#endif
