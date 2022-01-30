
#include "evidenceSeq.h"
#include "density.h"      // for psi, grad, hess
#include "tools.h"        // for lse(), vec2mat()
#include "epmgp.h"        // for ep()


// [[Rcpp::export]]
double approxlogml(arma::umat G, u_int b, arma::mat V, u_int J) {

    Gwish* graph = new Gwish(G, b, V); // instantiate Gwish object
    // Rcpp::Rcout << "graph initialized" << std::endl;

    // arma::mat samps = graph->sampleGW(J); // obtain samps from gw distr
	arma::mat samps = samplegw(J, graph); // obtain samps from gw distr

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
    double res = approxHelpSeq(z, u_star, samps_psi, graph);
    // Rcpp::Rcout << "computed hybrid-ep calculation" << std::endl;

    return res;
} // end approxlogml() function


double approxHelpSeq(arma::mat z, arma::vec uStar, arma::mat xy, Gwish* graph) {

    // TODO: fix the input so that we don't have to pass two matrices that are
    // essentially the same thing
    u_int D = graph->D;
	Tree* tree = new Tree(z, true);
    std::unordered_map<u_int, arma::vec>* pmap = tree->getPartition();
	std::unordered_map<u_int, arma::uvec>* leafRowMap = tree->getLeafRowMap();
	unsigned int nLeaves = tree->getLeaves();
    unsigned int d = tree->getNumFeats();

    std::unordered_map<u_int, arma::vec> candidates = findOptPoints(
		xy, *leafRowMap, nLeaves, uStar, D
	);

    return integratePartition(graph, candidates, *pmap, nLeaves);
} // end approxHelpSeq() function


arma::mat evalPsi(arma::mat samps, Gwish* graph) {
	u_int J = samps.n_rows;
	arma::mat psi_mat(J, 1, arma::fill::zeros);
	for (u_int j = 0; j < J; j++) {
		arma::vec u = arma::conv_to<arma::vec>::from(samps.row(j));
		psi_mat(j, 0) = psi_cpp(u, graph);
	}
	arma::mat psi_df = arma::join_rows( samps, psi_mat );
	// arma::mat psi_df = arma::join_rows( psi_mat, samps );
	return psi_df;
} // end evalPsi() function


double integratePartition(Gwish* graph,
	std::unordered_map<u_int, arma::vec> candidates,
	std::unordered_map<u_int, arma::vec> bounds,
	u_int K) {

	u_int D = graph->D;    // dimension of parameter space
	arma::vec log_terms(K, arma::fill::zeros);
	arma::vec G_k(K, arma::fill::zeros);
	arma::mat H_k(D, D, arma::fill::zeros);
	arma::mat H_k_inv(D, D, arma::fill::zeros);
	arma::vec lambda_k(D, arma::fill::zeros);
	arma::vec b_k(D, arma::fill::zeros);
	arma::vec m_k(D, arma::fill::zeros);
	arma::vec lb(D, arma::fill::zeros);
	arma::vec ub(D, arma::fill::zeros);
	arma::vec u_k(D, arma::fill::zeros);
	arma::vec candidate_k(D, arma::fill::zeros);

	double psi_k;
	arma::mat psi_mat(D, D, arma::fill::zeros);
	//arma::vec bounds_k;
	for (u_int k = 0; k < K; k++) {

		candidate_k = candidates[k];
		u_k = candidate_k.
			elem(arma::conv_to<arma::uvec>::from(arma::linspace(0, D-1, D)));
		// Rcpp::Rcout<< u_k << std::endl;
		psi_mat = vec2mat(u_k, graph);
		psi_k = candidate_k(D);

		H_k = hess_gwish(psi_mat, graph); // 11/9: using general hessian
		H_k_inv = arma::inv_sympd(H_k);
		lambda_k = grad_gwish(psi_mat, graph); // 11/9: using general gradient
		b_k = H_k * u_k - lambda_k;
		m_k = H_k_inv * b_k;

		lb = bounds[k].elem(arma::conv_to<arma::uvec>::from(
			arma::linspace(0, 2 * D - 2, D)));
		ub = bounds[k].elem(arma::conv_to<arma::uvec>::from(
			arma::linspace(1, 2 * D - 1, D)));

		double val = 0;
		double sign;
		log_det(val, sign, H_k);
		G_k(k) = ep(m_k, H_k_inv, lb, ub);
		log_terms(k) = D / 2 * std::log(2 * M_PI) - 0.5 * val - psi_k +
			arma::dot(lambda_k, u_k) -
			(0.5 * u_k.t() * H_k * u_k).eval()(0,0) +
			(0.5 * m_k.t() * H_k * m_k).eval()(0,0) + G_k(k);
	}

	return lse(log_terms, K);
} // end integratePartition() function


/* ----------------- older stuff -- old tree implementation ----------------- */

// [[Rcpp::export]]
double approxlogml_slow(arma::umat G, u_int b, arma::mat V, u_int J) {

    Gwish* graph = new Gwish(G, b, V); // instantiate Gwish object

    // arma::mat samps = graph->sampleGW(J); // obtain samps from gw distr
	arma::mat samps = samplegw(J, graph);

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
    double res = old_helper(z, u_star, samps_psi, graph);
    // Rcpp::Rcout << "computed hybrid-ep calculation" << std::endl;

    return res;
} // approxlogml_slow() function

double old_helper(arma::mat z, arma::vec uStar, arma::mat xy, Gwish* graph) {

    // TODO: fix the input so that we don't have to pass two matrices that are
    // essentially the same thing

    u_int D = graph->D;
    Tree* tree = new Tree(z); // passed with no boolean is slow tree algo
    std::unordered_map<u_int, arma::vec>* pmap = tree->getPartition();
	std::unordered_map<u_int, arma::uvec>* leafRowMap = tree->getLeafRowMap();
	unsigned int nLeaves = tree->getLeaves();
    unsigned int d = tree->getNumFeats();

    std::unordered_map<u_int, arma::vec> candidates = findOptPoints(
		xy, *leafRowMap, nLeaves, uStar, D
	);

    return integratePartition(graph, candidates, *pmap, nLeaves);

} // end old_helper() function


arma::mat samplegw(u_int J, Gwish* graph) {
	// have to convert otherwise compiler complains about unsigned int mat
    arma::mat G = arma::conv_to<arma::mat>::from(graph->G);
    arma::mat samps(graph->D, J, arma::fill::zeros);
    arma::mat omega, phi, zeta;
    arma::vec u0, u;
	arma::uvec ids  = graph->free_index;
    for (unsigned int j = 0; j < J; j++) {
        omega = rgwish_c(G, graph->P, graph->b, graph->p, 1e-8);
        phi   = arma::chol(omega);             // upper choleksy
        zeta  = phi * graph->P_inv;             // compute transformation
        u0    = arma::vectorise(zeta);
        u     = u0(ids);                       // extract free elements
        samps.col(j) = u;
    } // end sampling loop
     return samps.t(); // return as a (J x D) matrix
} // end samplegw() function


// end evidenceSeq.cpp file
