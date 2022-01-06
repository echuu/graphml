

#include "Graph.h"
#include "density.h"

// [[Rcpp::export]]
double test(arma::umat G, u_int b, arma::mat V, u_int J) {

    Graph* graph = new Graph(G, b, V); // instantiate Graph object
    // Rcpp::Rcout << "graph initialized" << std::endl;

    arma::mat samps = graph->sampleGW(J); // obtain sames from gw distribution
    // Rcpp::Rcout << "obtained g-wishart samples" << std::endl;

    // TODO: fix evalPsi so that we can use the z = [y | X] instead of [X | y]
    arma::mat samps_psi = evalPsi(samps, graph); // evaluate samples w/ psi()
    // Rcpp::Rcout << "evaluated samples" << std::endl; 

    // calculate global mode
    arma::vec u_star = calcMode(samps_psi, graph);
    // Rcpp::Rcout << "computed mode" << std::endl;
    // Rcpp::Rcout << u_star << std::endl;

    // format data so that it looks like z = [y | X]
    // format the data so that it looks like z = [y | X]
	arma::mat z = arma::join_rows( samps_psi.col(samps_psi.n_cols - 1), samps );

    // compute the final approximation
    double res = approxlogml(z, u_star, samps_psi, graph);
    // Rcpp::Rcout << "computed hybrid-ep calculation" << std::endl;

    return res;
}