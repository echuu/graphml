#ifndef PARTITION_H
#define PARTITION_H

arma::mat support(arma::mat samps, u_int D);
Rcpp::List fitTree(Rcpp::DataFrame x, Rcpp::Formula formula);
arma::vec findCandidatePoint(arma::mat data, arma::vec uStar, u_int dim);
std::unordered_map<int, arma::vec> findAllCandidatePoints(arma::mat data,
    arma::vec locs, arma::vec uStar, u_int D);
std::unordered_map<int, arma::vec> createPartitionMap(arma::mat bounds,
    arma::vec leafId);

#endif
