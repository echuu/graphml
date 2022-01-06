#ifndef NODE_H
#define NODE_H

#include <RcppArmadillo.h>

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
        
        Node(double threshold, unsigned int column, arma::mat data, 
            arma::uvec leftRows, arma::uvec rightRows);
        Node(double threshold, unsigned int column, arma::mat data, 
            arma::uvec leftRows, arma::uvec rightRows, double nodeSSE);
        Node(arma::mat data, arma::uvec leafRows);
        
        double calculateLeafValue(arma::mat data);
        void printNode();
        void setSSE(double nodeSSE);
        arma::uvec getLeafRows();
        unsigned int getFeature();
        double getThreshold();
        double getSSE();
        arma::mat getData();
        double getLeafVal();
}; // end Node class

#endif