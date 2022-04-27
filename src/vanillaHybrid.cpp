
#include "graphml_types.h"
#include "partition.h"      // for findOptPoints()
#include "Tree.h"           // for cart tree building algorithm
#include "tools.h"          // for lse(), vec2mat()
#include "epmgp.h"          // for ep()

// [[Rcpp::export]]
double hyb(arma::mat z) {
    // z = [ y | X ]
    u_int D = z.n_cols - 1; // dimension of the parameter
    Tree* tree = new Tree(z, true);
    std::unordered_map<u_int, arma::vec>* pmap = tree->getPartition();
	std::unordered_map<u_int, arma::uvec>* leafRowMap = tree->getLeafRowMap();
	unsigned int nLeaves = tree->getLeaves();

    Rcpp::Rcout << "Number of partition sets: " << nLeaves << std::endl;

    return 0;
} // end of hyb() function



