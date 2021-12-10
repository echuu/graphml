
#include "gwish.h"
#include "rgwishart.h"
#include "partition.h"
#include "tools.h"
#include "epmgp.h"
#include "gwishDensity.h"
#include <cmath>
#define RCPP_ARMADILLO_RETURN_COLVEC_AS_VECTOR

// [[Rcpp::depends(RcppArmadillo)]]
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

using namespace Rcpp;

/** ------------------------------------------------------------------------ **/


/* ------------------  algorithm specific functions ------------------------- */


// [[Rcpp::export]]
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


/* ----------------- evalute psi for each of the samples -------------------- */

/* ---------------------  general wrapper function  ------------------------- */

// generalApprox() <-- approx_v1() <-- approxZ()
// approx_integral() is the old function 


// [[Rcpp::export]]
double generalApprox(arma::umat G, u_int b, arma::mat V, u_int J) {

    // Rcpp::Rcout << p << " x " << p << " graph" << std::endl;
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
    return approx_v1(u_df,
                     u_star,
                     samps_psi,
                     obj);
} // end generalApprox() function


// [[Rcpp::export]]
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

    std::unordered_map<int, arma::vec> partitionMap;
    for (int i = 0; i < k; i++) {
        // add the leaf node's corresponding rectangle into the map
        int leaf_i = leafId(i);
        // arma::vec col_d = part[d];
        partitionMap[leaf_i] = part.col(i);
    }
	// -------------------------------------------------------------------------
	/* */


	std::unordered_map<int, arma::vec> candidates = findAllCandidatePoints(
		data, locs, uStar, D
	);

	// std::unordered_map<int, arma::vec> boundMap = partitionMap;
	 return approxZ(params, leafId, candidates, partitionMap, k);
	 
} // end approxZ() function


double approxZ(Rcpp::List& params,
	arma::vec leaf,
	std::unordered_map<int, arma::vec> candidates,
	std::unordered_map<int, arma::vec> bounds,
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
		G_k(k) = ep_logz(m_k, H_k_inv, lb, ub);
		log_terms(k) = D / 2 * std::log(2 * M_PI) - 0.5 * val - psi_k +
			arma::dot(lambda_k, u_k) -
			(0.5 * u_k.t() * H_k * u_k).eval()(0,0) +
			(0.5 * m_k.t() * H_k * m_k).eval()(0,0) + G_k(k);
	}

	return lse(log_terms, K);
} // end approxZ() function



/* ---------------------  approximation functions --------------------------- */

// [[Rcpp::export]]
double approx_integral(u_int K, arma::mat& psi_df, arma::mat& bounds,
	Rcpp::List& params) {

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

	for (u_int k = 0; k < K; k++) {

		// Rcpp::Rcout<< k << std::endl;

		u_k = arma::conv_to< arma::vec >::from(psi_df.submat(k, 0, k, D-1));

		arma::mat psi_mat = vec2mat(u_k, params);
		H_k = hess_gwish(psi_mat, params); // 11/9: using general hessian
		H_k_inv = inv(H_k);
		lambda_k = grad_gwish(psi_mat, params); // 11/9: using general gradient
		b_k = H_k * u_k - lambda_k;
		m_k = H_k_inv * b_k;

		// TODO: extract the lower and upper bounds of the k-th partition
		for (u_int d = 0; d < D; d++) {
			lb(d) = bounds.row(k)(2 * d);
			ub(d) = bounds.row(k)(2 * d + 1);
		}

		double val = 0;
		double sign;
		log_det(val, sign, H_k);
		// Rcpp::Rcout << val << std::endl;

		// TODO: load the epmgp code into the same directory so that we can use
		// the EP code directly without having to go back into R env
		G_k(k) = ep_logz(m_k, H_k_inv, lb, ub);

		// Rcpp::Rcout<< psi_df(k, D) << std::endl;

		log_terms(k) = D / 2 * std::log(2 * M_PI) - 0.5 * val - psi_df(k, D) +
			arma::dot(lambda_k, u_k) -
			(0.5 * u_k.t() * H_k * u_k).eval()(0,0) +
			(0.5 * m_k.t() * H_k * m_k).eval()(0,0) + G_k(k);
	} // end for() over k

	// TODO: find log-sum-exp function in arma
	return lse(log_terms, K);
} // end approx_integral() function



/** -------------------- end of implementation ----------------------------- **/


// end gwish.cpp file
