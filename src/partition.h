#ifndef PARTITION_H
#define PARTITION_H

arma::mat support(arma::mat samps, u_int D);
Rcpp::List fitTree(Rcpp::DataFrame x, Rcpp::Formula formula);
arma::vec findCandidatePoint(arma::mat data, arma::vec uStar, u_int dim);
std::unordered_map<u_int, arma::vec> findAllCandidatePoints(arma::mat data,
    arma::vec locs, arma::vec uStar, u_int D);
std::unordered_map<int, arma::vec> createPartitionMap(arma::mat bounds,
    arma::vec leafId);

// updated function to work with the cart functions
std::unordered_map<u_int, arma::vec> findOptPoints(arma::mat data,
    std::unordered_map<u_int, arma::uvec> leafRowMap,
    u_int numLeaves, arma::vec uStar, u_int D);

 arma::mat createDefaultPartition(arma::mat supp, u_int d, u_int k);

#endif
