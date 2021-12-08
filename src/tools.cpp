#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <cmath>
#define RCPP_ARMADILLO_RETURN_COLVEC_AS_VECTOR
// [[Rcpp::depends(RcppArmadillo)]]
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS


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

// [[Rcpp::export]]
std::unordered_map<int, arma::vec> getPartition(Rcpp::List tree, arma::mat supp) {

    Rcpp::Environment tmp = Rcpp::Environment::global_env();
    Rcpp::Function f = tmp["extractPartitionSimple"];
    
    Rcpp::List partList = f(tree, supp);
    arma::mat part = partList["partition"];
    // Rcpp::Rcout<< part << std::endl;
    arma::vec leafId = partList["leaf_id"];
    int k = leafId.n_elem; // # of leaf nodes

    arma::vec locs = partList["locs"];
    Rcpp::Rcout<< locs << std::endl;    

    std::unordered_map<int, arma::vec> partitionMap;
    for (int i = 0; i < k; i++) {
        // add the leaf node's corresponding rectangle into the map
        int leaf_i = leafId(i);
        // arma::vec col_d = part[d];
        partitionMap[leaf_i] = part.col(i);
    }

    // Rcpp::Function unname = tmp["unname"];
    // Rcpp::Rcout<< unname(tree["where"]) << std::endl;    

    return partitionMap;
    // store first column as leaf_id
    // store the remaining columns as the lb/ub of the partition sets
    // create the map that defines the boundary here
    
    // return part;
} // end getPartition() function















