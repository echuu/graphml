#ifndef DENSITY_H
#define DENSITY_H   

#include "graphml_types.h"
#include "Gwish.h"
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

/*  this file includes the implementations for the density, gradient, hessian 
    functions, which tend to be a bit messier. The goal of this file is to 
    replace the gwishDensity.cpp implementation by doing away with the 'param'
    argument which is an R object. In this implementation, we turn the density
    functions into member functions of the Gwish class so that we can directly
    access the variables we need instead of extracting them out of a parameter
    object. */


/** ---------------------- hybrid-ep functions ----------------------------- **/
arma::vec calcMode(arma::mat u_df, Gwish* graph);

/** ---------------------- objective functions ----------------------------- **/
double h(u_int i, u_int j, arma::mat& L);
arma::mat vec2mat(arma::vec u, Gwish* graph);
double psi_cpp(arma::vec& u, Gwish* graph);
double psi_cpp_mat(arma::mat& psi_mat, Gwish* graph);

/** ----------------------  gradient functions ----------------------------- **/
arma::vec grad_gwish(arma::mat& psi_mat, Gwish* graph);
double dpsi_ij(u_int i, u_int j, arma::mat& psi_mat, Gwish* graph);
double dpsi(u_int r, u_int s, u_int i, u_int j, arma::mat& psi, Gwish* graph);

/** ------------------------ hessian functions ----------------------------- **/
arma::mat hess_gwish(arma::mat& psi_mat, Gwish* graph);
double d2psi_ijkl(u_int i, u_int j, u_int k, u_int l, 
    arma::mat& psi, Gwish* graph);
double d2psi(u_int r, u_int s, u_int i, u_int j, u_int k, u_int l,
	arma::mat& psi, Gwish* graph);


#endif
