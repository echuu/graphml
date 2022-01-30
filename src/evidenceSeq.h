
#ifndef EVIDENCE_SEQ_H
#define EVIDENCE_SEQ_H

#include "graphml_types.h"
#include "Gwish.h"
#include "partition.h"  // for findOptPoints()
#include "Tree.h" 

double approxlogml(arma::umat G, u_int b, arma::mat V, u_int J);
double approxHelpSeq(arma::mat z, arma::vec uStar, arma::mat xy, Gwish* graph);
arma::mat evalPsi(arma::mat samps, Gwish* graph);
double integratePartition(Gwish* graph,
	std::unordered_map<u_int, arma::vec> candidates,
	std::unordered_map<u_int, arma::vec> bounds,
	u_int K);

/* older, slower implementations */
double approxlogml_slow(arma::umat G, u_int b, arma::mat V, u_int J);
double old_helper(arma::mat z, arma::vec uStar, arma::mat xy, Gwish* graph);

arma::mat samplegw(u_int J, Gwish* graph);


#endif