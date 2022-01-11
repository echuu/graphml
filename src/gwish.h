#ifndef GWISH_H
#define GWISH_H
#include <RcppArmadillo.h>
#include "graphml_types.h"
#include <Rcpp.h>
#include <vector>
#include "density.h"
// [[Rcpp::depends(RcppArmadillo)]]



Rcpp::List init_graph(arma::umat G, u_int b, arma::mat V);
Rcpp::List initTreeParams(u_int d);

std::vector<double> evalPsiParallel(arma::mat samps, Graph* graph);
arma::mat evalPsi(arma::mat samps, Graph* graph);


// approx_pll() <-- integratePartitionFast()

double integratePartitionFast(Graph* graph, 
	arma::vec leafId,
	std::unordered_map<u_int, arma::vec> candidates, 
	std::unordered_map<u_int, arma::vec> bounds, 
	u_int nLeaves);

double approx_pll(Rcpp::DataFrame u_df,
				 arma::vec uStar,
				 arma::mat data,
				 Graph* graph,
				 Rcpp::List& params);



#endif
