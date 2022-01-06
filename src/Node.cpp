
#include "Node.h"

// constructor for decision Node
Node::Node(double threshold, unsigned int column, arma::mat data, 
    arma::uvec leftRows, arma::uvec rightRows) {
    this->threshold  = threshold;
    this->column     = column;
    this->data       = data;
    this->leftRows   = leftRows;
    this->rightRows  = rightRows;
    this->leftCount  = leftRows.n_elem;
    this->rightCount = rightRows.n_elem;
}

// constructor for decision Node that also stores SSE of left/right
Node::Node(double threshold, unsigned int column, arma::mat data, 
    arma::uvec leftRows, arma::uvec rightRows, double nodeSSE) {
        this->threshold  = threshold;
        this->column     = column;
        this->data       = data;
        this->leftRows   = leftRows;
        this->rightRows  = rightRows;
        this->leftCount  = leftRows.n_elem;
        this->rightCount = rightRows.n_elem;
        this->nodeSSE    = nodeSSE;
}

// constructor for leaf Node
Node::Node(arma::mat data, arma::uvec leafRows) {
    this->leafRows = leafRows;
    this->data = data;
    this->isLeaf = true;
    this->leafVal = calculateLeafValue(data); // compute value of the leaf node
} 


double Node::calculateLeafValue(arma::mat data) {
    arma::vec y = data.col(0);
    return arma::mean(y);
}


void Node::printNode() {
    if (!this->isLeaf) {
        Rcpp::Rcout << "X" << this->column << " < " << 
            this->threshold << std::endl;
    } else {
        Rcpp::Rcout << "Leaf Node: " << this->leafVal << std::endl;
    }
}

/* ---------------------------- setters ----------------------------  */
void Node::setSSE(double nodeSSE) {
    this->nodeSSE = nodeSSE;
}

/* ---------------------------- getters ----------------------------  */
arma::uvec Node::getLeafRows() {
    return this->leafRows;
}
unsigned int Node::getFeature() {
    return this->column;
}
double Node::getThreshold() {
    return this->threshold;
}
double Node::getSSE() {
    return this->nodeSSE;
}
arma::mat Node::getData() {
    return this->data;
}
double Node::getLeafVal() {
    if (this->isLeaf) {
        return this->leafVal;
    }
    Rcpp::Rcout<< "no leaf value for decision node -- returning 0" << 
        std::endl;
    return 0;
}
/* ------------------------ end  getters ---------------------------  */

