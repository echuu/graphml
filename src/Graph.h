#ifndef GRAPH_H
#define GRAPH_H   


#include "graphml_types.h"
#include "rgwishart.h"

#include <cmath>

#define RCPP_ARMADILLO_RETURN_COLVEC_AS_VECTOR
// [[Rcpp::depends(RcppArmadillo)]]
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS


class Graph {

    public: 
        u_int p;               // # of vertices
        u_int D;               // # of free parameters
        u_int n_nonfree;       // # of non-free parameters
        u_int b;               // degrees of freedom for gwish density
        arma::umat G;          // adjaceny matrix for the graph G
        arma::mat V;           // scale matrix for gwish density
        arma::mat P;           // P = chol(inv(V)), where V is scale matrix
        arma::mat P_inv;       // P^(-1)
        arma::mat F;           // upper tri matrix of G, see A matrix in Atay
        arma::vec free;        // indicator for free elements
        arma::uvec free_index; // indices of the free elements
        arma::vec edgeInd;     // indicator for the UPPER diagonal free elements
        arma::vec k_i;         // gwish density parameter computed using rows(F)
        arma::vec nu_i;        // gwish density parameter computed using cols(F)
        arma::vec b_i;         // gwish density parameter computed using k, nu
        arma::mat t_ind;       // index matrix for free paramters
        arma::mat vbar;        // indices of nonfree elements

        Graph(arma::umat G, u_int b, arma::mat V);
        arma::mat getFreeElem();
        arma::mat getNonFreeElem();
        // arma::mat sampleGW(u_int m);
        // arma::mat sampleGWParallel(u_int J);
        

}; // end Graph class

#endif

