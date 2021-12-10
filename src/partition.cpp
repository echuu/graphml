#include <RcppArmadillo.h>
#include "graphml_types.h"
#include <Rcpp.h>
#include <cmath>
#define RCPP_ARMADILLO_RETURN_COLVEC_AS_VECTOR
// [[Rcpp::depends(RcppArmadillo)]]
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

/** ------------------------------------------------------------------------ **/


// [[Rcpp::export]]
arma::mat support(arma::mat samps, u_int D) {

    arma::mat s(D, 2, arma::fill::zeros);
    for (u_int d = 0; d < D; d++) {
        s(d, 0) = samps.col(d).min();
        s(d, 1) = samps.col(d).max();
    }
    return s;
} // end support() function


/*
    fitTree(): returns an tree object, makes call to rpart::rpart() --
    this eventually will need to be ported to C++, waiting for Donald to finish
    that implementation
*/
// [[Rcpp::export]]
Rcpp::List fitTree(Rcpp::DataFrame x, Rcpp::Formula formula) {

    // Obtain environment containing function
    Rcpp::Environment base("package:rpart");
    Rcpp::Environment stats("package:stats");
    // Make function callable from C++
    Rcpp::Function cart = base["rpart"];
    Rcpp::Function model_frame = stats["model.frame"];
    // Call the function and receive its list (confirm?) output
    Rcpp::DataFrame x_mat = model_frame(Rcpp::_["formula"] = formula,
                                        Rcpp::_["data"] = x);

    Rcpp::List tree = cart(x_mat);

    return tree;
} // end fitTree() function


arma::vec findCandidatePoint(arma::mat data, arma::vec uStar, u_int dim) {

    u_int D = dim;
    data = data.t(); // D x J_k, J_k = # of points in k-th partition

    u_int j_k = data.n_cols;
    arma::vec l1Cost(j_k); // store each of the distances
    arma::vec u;

    for (u_int j = 0; j < j_k; j++) {
        arma::vec u = data.col(j);
        u = u.elem(arma::conv_to<arma::uvec>::from(arma::linspace(0, D-1, D)));
        //Rcpp::Rcout<< arma::linspace(0, D-1) << std::endl;
        // Rcpp::Rcout<< u << std::endl;
        l1Cost(j) = arma::norm(u - uStar, 1);
    }
    // find index of the point that has the lowest l1-norm
    // *** NOTE: the (D+1)-th element
    arma::vec u_k = data.col(l1Cost.index_min());
    return u_k;
} // end findCandidatePoint() function


// [[Rcpp::export]]
std::unordered_map<int, arma::vec> findAllCandidatePoints(arma::mat data,
    arma::vec locs, arma::vec uStar, u_int D) {
    /* locs should correspond to the each of the data points */
    arma::vec locsUnique = arma::unique(locs);
    int n = locsUnique.n_elem;
    arma::mat data_subset;
    arma::uvec locRows;
    //arma::mat candidates(n, D, arma::fill::zeros); // store the candidate pts
    std::unordered_map<int, arma::vec> candidate_map;

    for (u_int i = 0; i < n; i++) {
        int loc = locsUnique(i);
        locRows = find(locs == loc);
        // Rcpp::Rcout<< locRows << std::endl;
        data_subset = data.rows(locRows);
        // Rcpp::Rcout<< findCandidatePoint(data_subset, uStar, D) << std::endl;
        // candidates.row(i) = findCandidatePoint(data_subset, uStar, D).t();
        candidate_map[loc] = findCandidatePoint(data_subset, uStar, D);
    }
    // candidate_map[2] = findCandidatePoint(data_subset, uStar, D);
    // Rcpp::Rcout<< candidate_map[2] << std::endl;

    return candidate_map;
} // end findAllCandidatePoints() function


// [[Rcpp::export]]
std::unordered_map<int, arma::vec> createPartitionMap(arma::mat bounds,
    arma::vec leafId) {

    // bounds is (2D) x K, where the lb/ub for each leaf node are stored col
    int k = leafId.n_elem; // # of leaf nodes

    std::unordered_map<int, arma::vec> partitionMap;
    for (u_int i = 0; i < k; i++) {
        // add the leaf node's corresponding rectangle into the map
        int leaf_i = leafId(i);
        partitionMap[leaf_i] = bounds.col(i);
    }

    return partitionMap;
} // end createPartitionMap() function


/* TODO: I dont' think we use this function anymore because I stuck it
directly into the approx_v1() function -- */
// [[Rcpp::export]]
Rcpp::List getPartition(Rcpp::List tree, arma::mat supp) {

    Rcpp::Environment tmp = Rcpp::Environment::global_env();
    Rcpp::Function f = tmp["extractPartitionSimple"];

    Rcpp::List partList = f(tree, supp);
    arma::mat part = partList["partition"];
    // Rcpp::Rcout<< part << std::endl;
    arma::vec leafId = partList["leaf_id"];
    int k = leafId.n_elem; // # of leaf nodes

    arma::vec locs = partList["locs"];
    // Rcpp::Rcout<< locs << std::endl;

    std::unordered_map<int, arma::vec> partitionMap;
    for (int i = 0; i < k; i++) {
        // add the leaf node's corresponding rectangle into the map
        int leaf_i = leafId(i);
        // arma::vec col_d = part[d];
        partitionMap[leaf_i] = part.col(i);
    }

    // Rcpp::Function unname = tmp["unname"];
    // Rcpp::Rcout<< unname(tree["where"]) << std::endl;

    return Rcpp::List::create(Rcpp::Named("locs") = locs,
						      Rcpp::Named("leafId") = leafId
            );
    // store first column as leaf_id
    // store the remaining columns as the lb/ub of the partition sets
    // create the map that defines the boundary here

    // return part;
} // end getPartition() function

// end partition.cpp 