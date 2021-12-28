
#ifndef INTERVAL_H
#define INTERVAL_H

#include "graphml_types.h"
#include "Node.h"
#include <vector>

struct Interval;
// arma::mat support(arma::mat samps, u_int D);
void dfs(Node* node, unsigned int& k, std::vector<Interval*> intervalStack,
    arma::mat& partition, arma::mat& supp, 
    std::unordered_map<u_int, arma::uvec>& leafRowMap);

struct Interval {
    public:
        double lb;
        double ub;
        unsigned int feature;
        Interval(double lb, double ub, unsigned int feature) {
            this->lb = lb;
            this->ub = ub;
            this->feature = feature;
        }
        void print() {
             Rcpp::Rcout << "[ " << this->lb << ", " << 
                this->ub << "]" << std::endl;
        }
};

// arma::mat support(arma::mat samps, u_int D) {
//     arma::mat s(D, 2, arma::fill::zeros);
//     for (u_int d = 0; d < D; d++) {
//         s(d, 0) = samps.col(d).min();
//         s(d, 1) = samps.col(d).max();
//     }
//     return s;
// } // end support() function

// recursive function to determine decision rules
void dfs(Node* node, unsigned int& k, std::vector<Interval*> intervalStack,
    arma::mat& partition, arma::mat& supp, 
    std::unordered_map<u_int, arma::uvec>& leafRowMap) {

    /*  k:          current partition set that we are updating
        decisions:  stack that holdes Interval objects
        partition:  nLeaves x (2 * nFeats) matrix, storing intervals row-wise
    */

    if (node->isLeaf) {
        /*
            extract the interval's feature
            replace the current interval values for that feature with the 
            updated interval values: lb = max(curr, new), ub = min(curr, new);
            this shrinks the interval to accommodate a tighter bound as we
            go deeper into the tree
        */
        for (const auto interval : intervalStack) {
            double lb = interval->lb;
            double ub = interval->ub; 
            unsigned int col = interval->feature; // features are 1-indexed

            // need to take max, min to accommodate for shrinking intervals,
            // further restrictions on the same feature as we go down the tree
            // partition(k, 2*col) = std::max(partition(k, 2*col), lb);
            // partition(k, 2*col+1) = std::min(partition(k, 2*col+1), ub);

            partition(2*col, k) = std::max(partition(2*col, k), lb);
            partition(2*col+1, k) = std::min(partition(2*col+1, k), ub);
        } // end of for() updating the k-th leaf

        // add the rows corresponding to this leaf into the hashmap
        // this is used in the hybrid algo to find candidate/representative 
        // points in each partition (leaf) set
        // NOTE: since we are doing this in the same order as storing the
        // intervals (above), we ensure that the k-th partition set
        // corresponds to the leafRows that we store in the map below
        leafRowMap[k] = node->getLeafRows();

        // Rcpp::Rcout << "row " << k << " : leaf value = " << node->getLeafVal() << std::endl;
        k++; // move to next leaf node / interval to update
        return;
    } // end of terminating condition

    /*  keep going down the tree and pushing the rules onto the stack until we
        reach a leaf node, then we use all elements in the stack to construct
        the interval. after constructing the interval, pop the leaf node off 
        so we can continue down another path to find the next leaf node
    */ 

    // extract the rule for the curr node -> this will give left and ride nodes
    // features are stored in the Node object as 1-index because the data 
    // looks like [y|X], so features start from column 1 rather than column 0
    // but when creating subsequent maps and matrices, we still want the 
    // first feature to be in the 0-th (first) column (element in map)
    unsigned int feature = node->getFeature() - 1;
    double lb = supp(feature, 0);
    double ub = supp(feature, 1);
    double threshVal = node->getThreshold();  // threshold value for node
    // construct interval for the left node
    Interval* leftInterval = new Interval(lb, threshVal, feature);
    // construct interval for the right node
    Interval* rightInterval = new Interval(threshVal, ub, feature);

    // travel down left node
    intervalStack.push_back(leftInterval);
    dfs(node->left, k, intervalStack, partition, supp, leafRowMap);
    delete intervalStack.back();
    intervalStack.pop_back();

    // travel down right node
    intervalStack.push_back(rightInterval);
    dfs(node->right, k, intervalStack, partition, supp, leafRowMap);
    delete intervalStack.back();
    intervalStack.pop_back();

} // end dfs() function





#endif
