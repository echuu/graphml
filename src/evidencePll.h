#ifndef EVIDENCE_PARALLEL_H
#define EVIDENCE_PARALLEL_H
#include <RcppArmadillo.h>
#include "graphml_types.h"
#include <vector>
#include "density.h"
// [[Rcpp::depends(RcppArmadillo)]]

double approxlogml_fast(arma::umat G, u_int b, arma::mat V, u_int J);

double approxHelpPll(arma::mat z, arma::vec uStar, arma::mat xy, Gwish* graph);

std::vector<double> evalPsiPll(arma::mat samps, Gwish* graph);

double integratePartitionPll(Gwish* graph, 
	std::unordered_map<u_int, arma::vec> candidates, 
	std::unordered_map<u_int, arma::vec> bounds, 
	u_int nLeaves);

arma::mat samplegwPll(u_int J, Gwish* graph);

#endif
