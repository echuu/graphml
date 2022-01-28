#ifndef GWISH_R_H
#define GWISH_R_H
#include "graphml_types.h"
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]



/** ------------------  algorithm specific functions  ---------------------- **/
Rcpp::List init_graph(arma::umat G, u_int b, arma::mat V);
arma::mat evalPsi(arma::mat samps, Rcpp::List& params);
arma::vec calcMode(arma::mat u_df, Rcpp::List& params);




/** ---------------------- R dataframe functions --------------------------- **/
Rcpp::StringVector createDfName(unsigned int D);
Rcpp::DataFrame mat2df(arma::mat x, Rcpp::StringVector nameVec);

/** -------------------- gwish-specific functions -------------------------- **/
arma::mat getFreeElem(arma::umat G, u_int p);
arma::mat getNonFreeElem(arma::umat G, u_int p, u_int n_nonfree);
double xi(u_int i, u_int j, arma::mat& L);



/** ----------- gwish density, gradient, hessian, root finding  ------------ **/

/** ---------------------  gwish density functions ------------------------- **/
double psi_cpp(arma::vec& u, Rcpp::List& params);
double psi_cpp_mat(arma::mat& psi_mat, Rcpp::List& params);


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






#endif
