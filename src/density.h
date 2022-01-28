#ifndef DENSITY_H
#define DENSITY_H   

#include "graphml_types.h"
#include "Graph.h"
#include <RcppArmadillo.h>
#include "gwish.h" // temporarily include for evalPsiParallel, move it out later
// [[Rcpp::depends(RcppArmadillo)]]

/*  this file includes the implementations for the density, gradient, hessian 
    functions, which tend to be a bit messier. The goal of this file is to 
    replace the gwishDensity.cpp implementation by doing away with the 'param'
    argument which is an R object. In this implementation, we turn the density
    functions into member functions of the Graph class so that we can directly
    access the variables we need instead of extracting them out of a parameter
    object. */


/* parallel */
// TODO: move parallel functions into its own file, leave this file to contain
// only functions that are common to both sequential and parallel implementations

double approxlogml_fast(arma::umat G, u_int b, arma::mat V, u_int J);
double integratePartitionFast(Graph* graph, 
	std::unordered_map<u_int, arma::vec> candidates, 
	std::unordered_map<u_int, arma::vec> bounds, 
	u_int nLeaves);

double approxlogml_map(arma::mat z, arma::vec uStar, arma::mat xy, Graph* graph);



/** ---------------------- hybrid-ep functions ----------------------------- **/
arma::vec calcMode(arma::mat u_df, Graph* graph);
// double approxlogml(arma::mat z, arma::vec uStar, arma::mat xy, Graph* graph);
// double integratePartition(Graph* graph, 
//     std::unordered_map<u_int, arma::vec> candidates, 
// 	std::unordered_map<u_int, arma::vec> bounds, 
// 	u_int nLeaves);

/** ---------------------- objective functions ----------------------------- **/
double h(u_int i, u_int j, arma::mat& L);
arma::mat vec2mat(arma::vec u, Graph* graph);
double psi_cpp(arma::vec& u, Graph* graph);
double psi_cpp_mat(arma::mat& psi_mat, Graph* graph);

/** ----------------------  gradient functions ----------------------------- **/
arma::vec grad_gwish(arma::mat& psi_mat, Graph* graph);
double dpsi_ij(u_int i, u_int j, arma::mat& psi_mat, Graph* graph);
double dpsi(u_int r, u_int s, u_int i, u_int j, arma::mat& psi, Graph* graph);

/** ------------------------ hessian functions ----------------------------- **/
arma::mat hess_gwish(arma::mat& psi_mat, Graph* graph);
double d2psi_ijkl(u_int i, u_int j, u_int k, u_int l, 
    arma::mat& psi, Graph* graph);
double d2psi(u_int r, u_int s, u_int i, u_int j, u_int k, u_int l,
	arma::mat& psi, Graph* graph);


#endif
