#ifndef TREE_H
#define TREE_H

#include "graphml_types.h"
#include "Interval.h"
#include "partition.h" // take out later when we put support() into Interval.h
#include <cmath>
#include "Node.h"


struct Interval;

class Tree {
    
    private:
        arma::uvec rowIds; // store row indices of the input
        arma::mat  z;      // data stored colwise w/ response in 1st col: [y|X]
        int numRows;       // number of data points given 
        int numFeats;      // dimension of feature (# of cols - 1)
        int k;             // number of leaves
        int numLeaves;
        double treeSSE;

        /* variables for checking terminating conditions */
        int nodeMinLimit;
        int minBucket;    // min number of elements in each leaf node
        double cp; 

        /* ---------- partition-related variables ---------- */
        arma::mat supp;
        arma::vec defInterval; 
        /* replace the role of partition matrix -- the partition matrix just
           gets converted into the partitionMap in the main approximation code
           so just create the partitionMap to start with to save conversion */
        std::unordered_map<u_int, arma::vec>* partitionMap; 
        std::unordered_map<u_int, arma::uvec>* leafRowMap;
        std::vector<Interval*>* intervalStack; 

        // map for storing feature -> sorted indices
        std::unordered_map<u_int, arma::uvec> featureIndexMap; 

        // member variables for the optimized version of tree building algorithm
        std::vector<u_int> yloc;

    public: 
        // TODO: make this private again after finish testing
        Node* root;      // root node for the tree (first split)
        double numSplits;
        
        Tree(arma::mat df);
        Tree(arma::mat df, bool fast);
        arma::uvec getSortedIndex(arma::mat data, arma::uvec rowvec, u_int d);
        Node* buildTreeSort(arma::uvec rowIds, double currSSE, u_int nodesize);
        Node* fastbuild(u_int nodenum, u_int start, u_int end, arma::uvec rowIds);
       
        void populateInterval(arma::vec& leafInterval);
        double calcImprove(arma::uvec left, arma::uvec right);
        double sse(arma::vec x, int n, double xbar);
        double calcSplitCp(double currSSE, double leftSSE, double rightSSE);
        double calculateSSE(arma::uvec rowIndex);
        arma::mat support(arma::mat X, u_int nfeats);
        
        int getLeaves();
        u_int getNumFeats();
        u_int getNumRows();
        double getSSE();
        std::unordered_map<u_int, arma::vec>* getPartition();
        std::unordered_map<u_int, arma::uvec>* getLeafRowMap();


        /* tree printing functions */
        void dfs(Node* ptr, std::string spacing = "");
        void printTree();

        /*  ------------------------- setters ------------------------------- */

}; 




#endif