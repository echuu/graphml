
#include "ogApprox.h"
#include "rgwishart.h"    // sampling from gw distribution

#include "partition.h"    // for support(), candidate point function
#include "epmgp.h"        // gaussian probability estimation
#include "tools.h"
#include "gwish_R.h"      // psi, grad, hess functions that take R object input

#include <Rcpp.h>
// [[Rcpp::depends(RcppArmadillo)]]

/*  ogApprox.cpp: Implementation for the hybrid-ep approximation that relies on
    rpart() from R to fit the tree. The fitted tree will be done in R and post-
    processed in C++ to extract the partition that we need for the integration
    scheme. 

    generalApprox() <-- approx_v1() <-- approxZ() 
*/

/* ----------------------- main 3 algorithm functions ----------------------- */

// [[Rcpp::export]]
double generalApprox(arma::umat G, u_int b, arma::mat V, u_int J) {
    // initialize graph object
    Rcpp::List obj = init_graph(G, b, V);
    // Rcpp::Rcout << "graph initialized" << std::endl;
    // generate J samples from gwishart
    arma::mat samps = rgw(J, obj);
    // Rcpp::Rcout << "obtained samples" << std::endl;
    // evalute the samples using the negative log posterior (psi in last column)
    arma::mat samps_psi = evalPsi(samps, obj);
    // Rcpp::Rcout << "evaluated samples" << std::endl;
    // convert samps_psi -> u_df_cpp (dataframe format) so that we can use CART
    Rcpp::DataFrame u_df = mat2df(samps_psi, obj["df_name"]); // in tools.cpp
    // Rcpp::Rcout << "convert to dataframe" << std::endl;
    // calculate global mode
    arma::vec u_star = calcMode(samps_psi, obj);
    // Rcpp::Rcout << "computed mode" << std::endl;

    // compute the final approximation
    return approx_v1(u_df, u_star, samps_psi, obj);
} // end generalApprox() function


double approx_v1(Rcpp::DataFrame u_df,
				 arma::vec uStar,
				 arma::mat data,
				 Rcpp::List& params) {

	u_int D = params["D"];
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
    arma::vec leafId = partList["leaf_id"];
    int k = leafId.n_elem; // # of leaf nodes
    arma::vec locs = partList["locs"];

    std::unordered_map<u_int, arma::vec> partitionMap;
    for (u_int i = 0; i < k; i++) {
        // add the leaf node's corresponding rectangle into the map
        u_int leaf_i = leafId(i);
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

	// std::unordered_map<int, arma::vec> boundMap = partitionMap;
	 return approxZ(params, leafId, candidates, partitionMap, k);
	 
} // end approxZ() function


double approxZ(Rcpp::List& params,
	arma::vec leaf,
	std::unordered_map<u_int, arma::vec> candidates,
	std::unordered_map<u_int, arma::vec> bounds,
	u_int K) {

	u_int D = params["D"];    // dimension of parameter space
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

	int leaf_k;
	double psi_k;
	arma::mat psi_mat(D, D, arma::fill::zeros);
	//arma::vec bounds_k;
	for (u_int k = 0; k < K; k++) {

		leaf_k = leaf(k);
		candidate_k = candidates[leaf_k];
		u_k = candidate_k.
			elem(arma::conv_to<arma::uvec>::from(arma::linspace(0, D-1, D)));
		// Rcpp::Rcout<< u_k << std::endl;
		psi_mat = vec2mat(u_k, params);
		// double psi_k = psi_cpp_mat(psi_mat, params);
		psi_k = candidate_k(D);

		H_k = hess_gwish(psi_mat, params); // 11/9: using general hessian
		// H_k_inv = inv(H_k);
		H_k_inv = arma::inv_sympd(H_k);
		lambda_k = grad_gwish(psi_mat, params); // 11/9: using general gradient
		b_k = H_k * u_k - lambda_k;
		m_k = H_k_inv * b_k;

		lb = bounds[leaf_k].elem(arma::conv_to<arma::uvec>::from(
			arma::linspace(0, 2 * D - 2, D)));
		ub = bounds[leaf_k].elem(arma::conv_to<arma::uvec>::from(
			arma::linspace(1, 2 * D - 1, D)));
		/*
		for (u_int d = 0; d < D; d++) {
			lb(d) = bounds[leaf_k](2 * d); ub(d) = bounds[leaf_k](2 * d + 1);
		}
		*/
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
} // end approxZ() function


/* ----------------------- algorithm helper functions ----------------------- */















