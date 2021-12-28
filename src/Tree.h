#ifndef TREE_H
#define TREE_H

#include "graphml_types.h"

#include "util.h"
#include <cmath>
#include "Node.h"

class Node;
Node* buildTree(arma::uvec rows);
double calculateRuleSSE(arma::uvec leftRows, arma::uvec rightRows);
bool isSmallerValue(double val);
int getLeaves();

class Tree {
    
    private:
        arma::uvec rowIds; // store row indices of the input
        arma::mat  z;      // data stored colwise w/ response in 1st col: [y|X]
        int numRows;       // number of data points given 
        int numFeats;      // dimension of feature (# of cols - 1)
        int k;             // number of leaves
        double minElem;    // min number of elements in each leaf node
        int numLeaves;

    public: 

        // TODO: make this private again after finish testing
        Node* root;      // root node for the tree (first split)

        Tree(arma::mat df) {
            this->z         = df;
            this->numRows   = df.n_rows;
            this->numFeats  = df.n_cols - 1;
            this->rowIds = arma::conv_to<arma::uvec>::from(
                    arma::linspace(0, this->numRows-1, this->numRows));
            this->minElem   = 5; // TODO: see min number in rpart() impl.
            this->numLeaves = 0;
            this->root      = buildTree(this->rowIds);

        } // end Tree() constructor

        // arma::uvec buildTree(arma::uvec rowIds) {
        Node* buildTree(arma::uvec rowIds) {
            // note: this function is called every time a tree is built on 
            // a child node; each output will give the root node of the tree

            int nodeMinLimit = 20;
            // check terminating conditions to break out of recursive calls
            if (rowIds.n_elem <= nodeMinLimit) { // TODO: add in cp check here
                this->numLeaves++;
                Node* leaf = new Node(this->z.rows(rowIds), rowIds);
                // Rcpp::Rcout<< "Leaf Node: " << leaf->getLeafVal() << std::endl;
                return leaf; // return pointer to leaf Node
            }

            // else: not at leaf node, rather a decision  node
            /* iterate through features, rows (data) to find (1) optimal 
               splitting feat and (2) optimal splitting value for that feature
            */
            double minSSE  = std::numeric_limits<double>::infinity();

            double optThreshold; // optimal split value
            u_int  optFeature;   // optimal feature to use in the split (col #)

            arma::uvec left, right;
            arma::uvec optLeft, optRight;
            u_int numFeats = this->numFeats;
            u_int numRows  = this->numRows;
            // since features are 1-indexed and numFeats features, this loop 
            // should go from 1 to numFeats (inclusive)
            for (u_int d = 1; d <= numFeats; d++) { // loop thru feats/cols
                // start from col 1 b/c y in col 0
                for (u_int n = 0; n < numRows; n++) { // loop thru data
                    // iterate through the rows corresponding to feature d 
                    // to find the optimal value on which to partition (bisect) 
                    // the data propose X(d, n) as the splitting rule

                    double threshold = z(n, d); // propose new splitting value
                    std::vector<u_int> rightRows, leftRows;

                    // construct the left/right row index vectors 
                    for (const auto &i : rowIds) {
                        // compare val with each of the vals in other rows
                        double val_i = z(i, d);
                        if (val_i <= threshold) { 
                            // equality guarantees that at least 1 node will be 
                            // less than the threshold.. is this what we want? 
                            // if we want a case where we allow for 0 nodes in 
                            // threshold just have the inequality be strict
                            
                            // determine right child vs. left child membership 
                            // based on value compared to threshold
                            leftRows.push_back(i);
                        } else {
                            rightRows.push_back(i);
                        }

                    } // end for() creating the left/right row index vectors

                    if (rightRows.size() == 0 || leftRows.size() == 0) {
                        // we do this check because we already checked the 
                        // condition of a leaf node in beginning of function
                        continue; // go back to start of INNER for over the rows
                    } 

                    // compute SSE associated with this decision rule
                    // convert rightRow, leftRow into arma::uvec
                    left  = arma::conv_to<arma::uvec>::from(leftRows);
                    right = arma::conv_to<arma::uvec>::from(rightRows);

                    double propSSE = calculateRuleSSE(left, right);
                    // Rcpp::Rcout<< "left: " << left << std::endl;
                    // Rcpp::Rcout<< "right: " << right << std::endl;
                    // Rcpp::Rcout<< "sse: " << propSSE << std::endl;
                    
                    // TODO: isn't cp parameter in rpart measuring how much the 
                    // improvement is? in that case, we should look at the diff
                    // in the MSE rather than if we can find a smaller one
                    if (propSSE < minSSE) {
                        // Rcpp::Rcout<< "enter if" << std::endl;
                        minSSE = propSSE;
                        // TODO: store threshold value; this defines partition
                        optThreshold = threshold;
                        // TODO: store col number b/c this gives partition #
                        // don't need to store the row (n) b/c we only need
                        // the value (threshold) and the feature (d) so we know
                        // where to define this 1-d interval in the partition
                        optFeature = d;
                        optLeft = left;
                        optRight = right;
                    } else {
                        // ignore the rule
                        // if we go the route of making a Rule object, 
                        // we'll need to delete the rule here
                    } // end if-else checking the new SSE
                    
                } // end for() over data, over z(i, d), for i in [0, nRows]

            } // end for() over features ------ end OUTER FOR LOOP()

            // ************************************************************
            // IMPORTANT:
            // nalds code has part here which constructs the left/right rows, 
            // but i  think that;s because his rule doesnt store these; 
            // since we have these, we dont need to reconstruct them. 
            // just put the definitions outside of main loops
            // ************************************************************

            // DEBUG CODE TO LOOK AT VALUES:
            // Rcpp::Rcout<< "optimal feature: " << optFeature << " ("
            //     << optThreshold << ")" << std::endl;
            // Rcpp::Rcout<< "optimal split val: " << optThreshold << std::endl;
            // Rcpp::Rcout<< "min SSE: " << minSSE << std::endl;

            int leftCount = optLeft.n_elem;
            int rightCount = optRight.n_elem;

            // construct node using optimal value, column, data, left, right
            Node* node = new Node(optThreshold, optFeature, z, optLeft, optRight);
            // Rcpp::Rcout << "# in left: " << node->leftCount << std::endl;
            // Rcpp::Rcout << "# in right: " << node->rightCount << std::endl;
            // Rcpp::Rcout<< "L: " << node->leftCount << " / R: " << 
            //     node->rightCount << std::endl;
            // Rcpp::Rcout<< "--------------------------------------" << std::endl;

            // build left and right nodes
            // in order to test one iteration of this function, just take out
            // left, right subtree function calls
            node->left  = buildTree(optLeft);
            node->right = buildTree(optRight);

            return node;
        } // end of buildTree() function

        // compute the SSE associated with this decision node (rule)
        double calculateRuleSSE(arma::uvec leftRows, arma::uvec rightRows) {
            arma::vec y = this->z.col(0);
            // subset out y-values for left node
            arma::vec yLeft = y.elem(leftRows); 
            // subset out y-values for right node
            arma::vec yRight = y.elem(rightRows);
            // compute left-mean
            double leftMean = arma::mean(yLeft);
            // compute right-mean
            double rightMean = arma::mean(yRight);

            // Rcpp::Rcout<< "leftMean: " << leftMean << std::endl;
            // Rcpp::Rcout<< "rightMean: " << rightMean << std::endl;
            int nLeft = yLeft.n_elem;
            int nRight = yRight.n_elem;
            // compute sse for left
            double sseLeft = sse(yLeft, nLeft, leftMean);
            // compute sse for right
            double sseRight = sse(yRight, nRight, rightMean);
            // compute node sse
            // Rcpp::Rcout<< "sseLeft: " << sseLeft << std::endl;
            // Rcpp::Rcout<< "sseRight: " << sseRight << std::endl;
            return sseLeft + sseRight;
        } // end calculateRuleSSE() function

        // compare this decision node's threshold value to other feature value
        bool isSmallerValue(double val) {
            // return val < this->threshold;
            return 0;
        } // end isSmallerValue() function

        void dfs(Node* ptr, std::string spacing = "") {
            Rcpp::Rcout << spacing;
            ptr->printNode();
            if (ptr->left) {
                Rcpp::Rcout << spacing << "---> true";
                dfs(ptr->left, spacing + " ");
            }
            if (ptr->right) {
                Rcpp::Rcout << spacing << "---> false";
                dfs(ptr->right, spacing + " ");
            }
        }

        void printTree() {
            dfs(root);
        }

        /*  ------------------------- getters ------------------------------- */
        int getLeaves() {
            return this->numLeaves;
        }
        u_int getNumFeats() {
            return this->numFeats;
        }
        u_int getNumRows() { 
            return this->numRows;
        }

        /*  ------------------------- setters ------------------------------- */

}; 



#endif