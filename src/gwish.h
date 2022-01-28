#ifndef GWISH_H
#define GWISH_H
#include <RcppArmadillo.h>
#include "graphml_types.h"
#include <Rcpp.h>
#include <vector>
#include "density.h"
// [[Rcpp::depends(RcppArmadillo)]]



std::vector<double> evalPsiParallel(arma::mat samps, Graph* graph);




#endif
