
#include "gwish.h"			// this file includes density.h
#include "rgwishart.h"
#include "partition.h"
#include "tools.h"
#include "epmgp.h"
#include "gwishDensity.h"
#include <cmath>

#include "Node.h"
#include "Graph.h"
#include "Tree.h"
#include "Interval.h"
#define RCPP_ARMADILLO_RETURN_COLVEC_AS_VECTOR

// [[Rcpp::depends(RcppArmadillo)]]
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

using namespace Rcpp;

/** ------------------------------------------------------------------------ **/

/* ------------------  algorithm specific functions ------------------------- */

Rcpp::List init_graph(arma::umat G, u_int b, arma::mat V) {

    // Rcpp::Rcout << G << std::endl;
	u_int p = G.n_rows;
	arma::mat P = chol(inv(V));
	arma::mat P_inv = arma::inv(P);
	arma::mat F = getFreeElem(G, p);          // upper triangular matrix of G
	arma::vec free = vectorise(F);            // indicator for free elements
	arma::uvec free_ids = find(free); // extract indices of the free elements
	arma::uvec upInd = trimatu_ind(size(G));

	// indicator for upper diag free
	arma::vec edgeInd = arma::conv_to<arma::vec>::from(G(upInd));
	// Rcpp::Rcout << "edge indicator" << std::endl;

	// construct A matrix to compute k_i
	arma::vec k_i  = arma::conv_to<arma::vec>::from(arma::sum(F, 0) - 1);
	arma::vec nu_i = arma::conv_to<arma::vec>::from(arma::sum(F, 1) - 1);
	arma::vec b_i  = nu_i + k_i + 1;
	u_int D = arma::sum(edgeInd);

	// create index matrix for the free parameters
	arma::uvec ind_vec = find(F > 0); // extract indices of the free elements
	arma::mat t_ind(D, 2, arma::fill::zeros);
	for (u_int d = 0; d < D; d++) {
		t_ind(d, 0) = ind_vec(d) % p; // row of free elmt
		t_ind(d, 1) = ind_vec(d) / p; // col of free elmt
	}
	/* can eventually remove this, but the other functions already subtract one
		 because they assume R implementation (i.e., 1-index instead of 0-index)
		 once everything is written in C++, we can fix the other functions and
		 remove the following line
	 */
	t_ind = t_ind + 1;
    // Rcpp::Rcout << t_ind << std::endl;

	u_int n_nonfree = p * (p + 1) / 2 - D; // # of nonfree elements
	arma::mat vbar = getNonFreeElem(G, p, n_nonfree);
    // Rcpp::Rcout << vbar << std::endl;

	Rcpp::Environment stats("package:stats");
	Rcpp::Function asFormula = stats["as.formula"];

	return List::create(Named("G") = G, Named("b") = b, Named("V") = V,
						Named("p") = p, Named("P") = P, Named("D") = D,
						Named("P_inv") = P_inv,
						Named("FREE_PARAMS_ALL") = free,
						Named("free_index") = free_ids,
						Named("edgeInd") = edgeInd,
						Named("k_i") = k_i,
						Named("nu_i") = nu_i,
						Named("b_i") = b_i,
						Named("t_ind") = t_ind,
						Named("n_nonfree") = n_nonfree,
						Named("vbar") = vbar,
						Named("df_name") = createDfName(D), // in tools.cpp
						Named("formula") = asFormula("psi_u ~.")
                      );
} // end init_graph() function


Rcpp::List initTreeParams(u_int d) {
	Rcpp::Environment stats("package:stats");
	Rcpp::Function asFormula = stats["as.formula"];
	return List::create(
		Named("df_name") = createDfName(d), // in tools.cpp
		Named("formula") = asFormula("psi_u ~.")
	);
}


std::vector<double> evalPsiParallel(arma::mat samps, Graph* graph) {
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
} // end evalPsi() function


arma::mat evalPsi(arma::mat samps, Graph* graph) {
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



double integratePartitionFast(Graph* graph, 
	arma::vec leafId,
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
    #pragma omp parallel shared(candidates, graph, bounds, nLeaves, leafId)
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
		u_int leaf_k;

        #pragma omp for // fill vec_private in parallel
        for (u_int k = 0; k < nLeaves; k++) {
			leaf_k = leafId(k);
            candidate_k = candidates[leaf_k];
            u_k = candidate_k.elem(uIndex);
            psi_mat = vec2mat(u_k, graph);
            psi_k = candidate_k(D);

            H_k = hess_gwish(psi_mat, graph); 
            H_k_inv = arma::inv_sympd(H_k);
            lambda_k = grad_gwish(psi_mat, graph); 
            b_k = H_k * u_k - lambda_k;
            m_k = H_k_inv * b_k;

            lb = bounds[leaf_k].elem(lbInd);
            ub = bounds[leaf_k].elem(ubInd);
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
} // end integratePartitionFast() function


double approx_pll(Rcpp::DataFrame u_df,
				 arma::vec uStar,
				 arma::mat data,
				 Graph* graph,
				 Rcpp::List& params) {

	u_int D = graph->D;
	Rcpp::Formula formula = params["formula"];
	// fit CART model
	Rcpp::List tree = fitTree(u_df, formula);
	// get the support
	arma::mat supp = support(data, D);

	/* extract partition from the rpart object -> see f() function call below;
	   ideally, we have a separate C++ function that extracts the partition
	   and turns it into the partitionMap object that we have below. */
	// -------------------------------------------------------------------------
	Rcpp::Environment tmp = Rcpp::Environment::global_env();
    Rcpp::Function f = tmp["extractPartitionSimple"];

    Rcpp::List partList = f(tree, supp);
    arma::mat part = partList["partition"];
    arma::vec leafId = partList["leaf_id"]; // set of leaf ids
    int k = leafId.n_elem; // # of leaf nodes
	 // n/j-dimensional, contains leaf id for each data point
    arma::vec locs = partList["locs"];

    std::unordered_map<u_int, arma::vec> partitionMap;
    for (u_int i = 0; i < k; i++) {
        // add the leaf node's corresponding rectangle into the map
        u_int leaf_i = leafId(i);
		// Rcpp::Rcout << "leaf id = " << leaf_i << std::endl;
        // arma::vec col_d = part[d];
        partitionMap[leaf_i] = part.col(i);
    }
	// -------------------------------------------------------------------------

	// go into here and figure how to use the ROWS of each partition set's 
	// points instead of finding the rows' locations that are equal to
	// the current leaf node / location that we're working on
	std::unordered_map<u_int, arma::vec> candidates = findAllCandidatePoints(
		data, locs, uStar, D
	);

	return integratePartitionFast(graph, leafId, candidates, partitionMap, k);
	///return integratePartition(graph, leafId, candidates, partitionMap, k);
}



/* ---------------  end of algorithm specific functions --------------------- */











// end gwish.cpp file
