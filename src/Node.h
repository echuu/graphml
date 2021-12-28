#ifndef NODE_H
#define NODE_H

#include <RcppArmadillo.h>

double calculateLeafValue(arma::mat data);

/* getters */
arma::uvec getLeafRows();
double getThreshold();
double getSSE();
arma::mat getData();
double getLeafVal();
void printNode();

void setSSE(double nodeSSE);

class Node {
    private:
        /* fields common to both decision nodes and leaf nodes */
        arma::mat     data;      // y in 1st column, X in 2nd-dth column
    
        double        threshold; // value used to determine left/right split
        double        nodeSSE;   // sum of squared error associated with split
        unsigned int  column;    // column of the splitting variable
        arma::uvec    leftRows;  // rows of data that go into left node
        arma::uvec    rightRows; // rows of data taht go into right node
        
        

        /* fields only defined for leaf nodes */
        double        leafVal;   // value for leaf, if is leaf
        arma::uvec    leafRows; 

    public:
        /* fields only defined for decision nodes */
        Node* left   = nullptr; // left child
        Node* right  = nullptr; // right child
        bool  isLeaf = false;   // indicator for leaf nodes
        int leftCount, rightCount;
        
        // constructor for Node
        Node(double threshold, unsigned int column, arma::mat data, 
            arma::uvec leftRows, arma::uvec rightRows) {
                this->threshold  = threshold;
                this->column     = column;
                this->data       = data;
                this->leftRows   = leftRows;
                this->rightRows  = rightRows;
                this->leftCount  = leftRows.n_elem;
                this->rightCount = rightRows.n_elem;
        }
        // constructor for leaf Node
        Node(arma::mat data, arma::uvec leafRows) {
            this->leafRows = leafRows;
            this->data = data;
            this->isLeaf = true;
            this->leafVal = calculateLeafValue(data); 
        }
        // compute the value of the leaf node
        double calculateLeafValue(arma::mat data) {
            arma::vec y = data.col(0);
            return arma::mean(y);
        }

        void printNode() {
            if (!this->isLeaf) {
                Rcpp::Rcout << "X" << this->column << " < " << 
                    this->threshold << std::endl;
            } else {
                Rcpp::Rcout << "Leaf Node: " << this->leafVal << std::endl;
            }
        }

        /* ---------------------------- setters ----------------------------  */
        void setSSE(double nodeSSE) {
            this->nodeSSE = nodeSSE;
        }

        /* ---------------------------- getters ----------------------------  */
        arma::uvec getLeafRows() {
            return this->leafRows;
        }
        u_int getFeature() {
            return this->column;
        }
        double getThreshold() {
            return this->threshold;
        }
        double getSSE() {
            return this->nodeSSE;
        }
        arma::mat getData() {
            return this->data;
        }
        double getLeafVal() {
            if (this->isLeaf) {
                return this->leafVal;
            }
            Rcpp::Rcout<< "no leaf value for decision node -- returning 0" << 
                std::endl;
            return 0;
        }
        /* ------------------------ end  getters ---------------------------  */

}; // end Node class

#endif