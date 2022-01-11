

#include "Graph.h"
#include "density.h"
#include "gwish.h"
#include "tools.h"
// bottom 2 for new testing that uses R code


// [[Rcpp::export]]
double testParallel(arma::umat G, u_int b, arma::mat V, u_int J) {

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
    double res = approxlogmlFast(z, u_star, samps_psi, graph);
    // Rcpp::Rcout << "computed hybrid-ep calculation" << std::endl;

    return res;
}


// [[Rcpp::export]]
double testmap(arma::umat G, u_int b, arma::mat V, u_int J) {

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
    double res = approxlogml_map(z, u_star, samps_psi, graph);
    // Rcpp::Rcout << "computed hybrid-ep calculation" << std::endl;

    return res;
}


// [[Rcpp::export]]
double approxfaster(arma::umat G, u_int b, arma::mat V, u_int J) {

    Graph* graph = new Graph(G, b, V); // instantiate Graph object
	Rcpp::List obj = initTreeParams(graph->D);

	// arma::mat samps = sampleGWParallel(J, graph);
	arma::mat samps = graph->sampleGWParallel(J);

	std::vector<double> psivec = evalPsiParallel(samps, graph);
	// std::vector<double> psivec = this->evalPsiParallel(samps, graph);
	arma::mat psi_col = arma::conv_to<arma::mat>::from(psivec);
	arma::mat samps_psi = arma::join_rows( samps, psi_col );

    Rcpp::DataFrame u_df = mat2df(samps_psi, obj["df_name"]); // in tools.cpp

    // calculate global mode
    arma::vec u_star = calcMode(samps_psi, graph);

    // compute the final approximation
    double res = approx_pll(u_df, u_star, samps_psi, graph, obj);

    return res;
}




