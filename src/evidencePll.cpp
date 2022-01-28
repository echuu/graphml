
#include "evidencePll.h"   // this file includes density.h : psi_cpp, grad, hess
#include "Graph.h"         // includes rgwishart.h <- rgwish_c()
#include "Tree.h"          // includes Node.h, partition.h, Interval.h, <cmath>
// #include "rgwishart.h"  // rgwish_c()
// #include "partition.h"
// #include "Node.h"
// #include "Interval.h"
#include "tools.h"         // lse()
#include "epmgp.h"         // ep()



#define RCPP_ARMADILLO_RETURN_COLVEC_AS_VECTOR

// [[Rcpp::depends(RcppArmadillo)]]
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

// parallelization:
#include <omp.h>
// [[Rcpp::plugins(openmp)]]

// 4 main functions
// approx()
// approx_helper()
// evalPsi()
// integratePartition()



// [[Rcpp::export]]
double approxlogml_fast(arma::umat G, u_int b, arma::mat V, u_int J) {

    Graph* graph = new Graph(G, b, V); // instantiate Graph object

	arma::mat samps = samplegwPll(J, graph); // (J x D); samps filled row-wise
	// arma::mat samps = graph->sampleGWParallel(J);

	std::vector<double> psivec = evalPsiPll(samps, graph);
	arma::mat psi_col = arma::conv_to<arma::mat>::from(psivec);
	arma::mat samps_psi = arma::join_rows( samps, psi_col );

    // calculate global mode
    arma::vec u_star = calcMode(samps_psi, graph);

    arma::mat z = arma::join_rows( samps_psi.col(samps_psi.n_cols - 1), samps );

    // compute the final approximation
    double res = approxHelpPll(z, u_star, samps_psi, graph);

    return res;
} // end approxlogml_fast() function


double approxHelpPll(arma::mat z, arma::vec uStar, arma::mat xy, Graph* graph) {

    // TODO: fix the input so that we don't have to pass two matrices that are
    // essentially the same thing

    u_int D = graph->D;
    // fit cart model 
	Tree* tree = new Tree(z, true);
    std::unordered_map<u_int, arma::vec>* pmap = tree->getPartition();
	std::unordered_map<u_int, arma::uvec>* leafRowMap = tree->getLeafRowMap();
	unsigned int nLeaves = tree->getLeaves();
    unsigned int d = tree->getNumFeats();

    std::unordered_map<u_int, arma::vec> candidates = findOptPoints(
		xy, *leafRowMap, nLeaves, uStar, D
	);

    return integratePartitionPll(graph, candidates, *pmap, nLeaves);

} // end approxHelpPll() function


std::vector<double> evalPsiPll(arma::mat samps, Graph* graph) {
	u_int J = samps.n_rows;
	std::vector<double> vec;
	size_t *prefix;
	#pragma omp parallel shared(samps)
    {	
		int ithread  = omp_get_thread_num();
		int nthreads = omp_get_num_threads();
		#pragma omp single
		{
			prefix = new size_t[nthreads+1];
			prefix[0] = 0;
		}
		std::vector<double> vec_private;
		#pragma omp for schedule(static) nowait
		for (u_int j = 0; j < J; j++) {
			arma::vec u = arma::conv_to<arma::vec>::from(samps.row(j));
			vec_private.push_back( psi_cpp(u, graph) );
		}
		prefix[ithread+1] = vec_private.size();
		#pragma omp barrier
		#pragma omp single 
		{
			for(int i=1; i<(nthreads+1); i++) prefix[i] += prefix[i-1];
			vec.resize(vec.size() + prefix[nthreads]);
		}
		std::copy(vec_private.begin(), vec_private.end(), 
			vec.begin() + prefix[ithread]);
	}
	return vec;
} // end evalPsiPll() function


// parallel integration routine
double integratePartitionPll(Graph* graph, 
	std::unordered_map<u_int, arma::vec> candidates, 
	std::unordered_map<u_int, arma::vec> bounds, 
	u_int nLeaves) {

    u_int D = graph->D;    // dimension of parameter space
    arma::uvec lbInd = arma::conv_to<arma::uvec>::from(
                arma::linspace(0, 2 * D - 2, D));
    arma::uvec ubInd = arma::conv_to<arma::uvec>::from(
                arma::linspace(1, 2 * D - 1, D));
    arma::uvec uIndex = arma::conv_to<arma::uvec>::from(
                arma::linspace(0, D-1, D));

    std::vector<double> vec; 
    // std::vector<int> iterations(omp_get_max_threads(), 0);
    #pragma omp parallel shared(candidates, graph, bounds, nLeaves)
    {
        // std::vector<double> vec_private (nLeaves);
        std::vector<double> vec_private;
        arma::mat H_k(D, D, arma::fill::zeros);
        arma::mat H_k_inv(D, D, arma::fill::zeros);
        arma::vec lambda_k(D, arma::fill::zeros);
        arma::vec b_k(D, arma::fill::zeros);
        arma::vec m_k(D, arma::fill::zeros);
        arma::vec lb(D, arma::fill::zeros);
        arma::vec ub(D, arma::fill::zeros);
        arma::vec u_k(D, arma::fill::zeros);
        arma::vec candidate_k(D, arma::fill::zeros);
        double psi_k, G_k, tmp;
	    arma::mat psi_mat(D, D, arma::fill::zeros);

        #pragma omp for // fill vec_private in parallel
        for (u_int k = 0; k < nLeaves; k++) {
            candidate_k = candidates[k];
            u_k = candidate_k.elem(uIndex);
            psi_mat = vec2mat(u_k, graph);
            psi_k = candidate_k(D);

            H_k = hess_gwish(psi_mat, graph); 
            H_k_inv = arma::inv_sympd(H_k);
            lambda_k = grad_gwish(psi_mat, graph); 
            b_k = H_k * u_k - lambda_k;
            m_k = H_k_inv * b_k;

            lb = bounds[k].elem(lbInd);
            ub = bounds[k].elem(ubInd);
            double val = 0;
            double sign;
            log_det(val, sign, H_k);
            G_k = ep(m_k, H_k_inv, lb, ub);
            tmp = D / 2 * std::log(2 * M_PI) - 0.5 * val - psi_k +
                arma::dot(lambda_k, u_k) -
                (0.5 * u_k.t() * H_k * u_k).eval()(0,0) +
                (0.5 * m_k.t() * H_k * m_k).eval()(0,0) + G_k; 
            vec_private.push_back(tmp);
            // iterations[omp_get_thread_num()]++;
        } // end of for loop

        #pragma omp critical
        vec.insert(vec.end(), vec_private.begin(), vec_private.end());
    } // end of omp

    // for (unsigned int j = 0; j < iterations.size(); j++) {
    //     Rcpp::Rcout << iterations[j] << std::endl;
    // }
	
	return lse(vec, nLeaves);
} // end integratePartitionPll() function


// samplegwPll(): generate J samples from gwish distribution in parallel
arma::mat samplegwPll(u_int J, Graph* graph) {
	// have to convert otherwise compiler complains about unsigned int mat
	arma::mat G = arma::conv_to<arma::mat>::from(graph->G);
	arma::mat samps(J, graph->D, arma::fill::zeros);
	arma::mat P_inv = graph->P_inv;

	std::vector<arma::vec> vec; 
	#pragma omp parallel shared(G)
    {
		arma::mat omega, phi, zeta;
		arma::vec u0, u;
		arma::uvec ids  = graph->free_index;
		std::vector<arma::vec> vec_private; 
		#pragma omp for // fill vec_private in parallel
		for (unsigned int j = 0; j < J; j++) {
			omega = rgwish_c(G, graph->P, graph->b, graph->p, 1e-8);
			phi   = arma::chol(omega);             // upper choleksy
			zeta  = phi * P_inv;                   // compute transformation
			u0    = arma::vectorise(zeta);
			u     = u0(ids);                       // extract free elements
			vec_private.push_back(u);
		} // end sampling loop
		#pragma omp critical
        vec.insert(vec.end(), vec_private.begin(), vec_private.end());
	} // end of outer omp

	for (unsigned int j = 0; j < J; j++) {
		arma::rowvec u = arma::conv_to< arma::rowvec >::from( vec[j] );
		samps.row(j) = u;
	}
	 return samps; // return (J x D) matrix, samps matrix filled row-wise
} // end samplegwPll() function


// end evidencePll.cpp file
