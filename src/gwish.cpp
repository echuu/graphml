
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
// [[Rcpp::export]]
arma::mat evalPsi(arma::mat samps, Rcpp::List& params) {
	u_int D = params["D"];
	u_int J = samps.n_rows;

	arma::mat psi_mat(J, 1, arma::fill::zeros);
	for (u_int j = 0; j < J; j++) {
		arma::vec u = arma::conv_to<arma::vec>::from(samps.row(j));
		psi_mat(j, 0) = psi_cpp(u, params);
	}

	arma::mat psi_df = arma::join_rows( samps, psi_mat );

	return psi_df;
} // end evalPsi() function


/* ---------------------  general wrapper function  ------------------------- */
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
}

// [[Rcpp::export]]
double generalApprox(arma::umat G, u_int b, arma::mat V, u_int J) {

    int p = G.n_rows;
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
		psi_mat = create_psi_mat_cpp(u_k, params);
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


/* -------------------  newton's method implementation ---------------------- */

// [[Rcpp::export]]
arma::vec calcMode(arma::mat u_df, Rcpp::List& params) {

		 double tol = 1e-8;
		 u_int maxSteps = 10;
		 bool VERBOSE = false;

		 // use the MAP as starting point for algorithm
		 u_int D = params["D"];
		 u_int mapIndex = u_df.col(D).index_min();

		 arma::vec theta = arma::conv_to<arma::vec>::from(u_df.row(mapIndex)).
		 	subvec(0, D-1);

		 u_int numSteps = 0;
		 double tolCriterion = 100;
		 double stepSize = 1;

		 arma::mat thetaMat = create_psi_mat_cpp(theta, params);
		 arma::mat thetaNew;
		 arma::mat thetaNewMat;
		 arma::mat G;
		 arma::mat invG;
		 double psiNew = 0, psiCurr = 0;
		 /* start newton's method loop */
		 while ((tolCriterion > tol) && (numSteps < maxSteps)) {
			 // thetaMat = create_psi_mat_cpp(theta, params);
			 // G = -hess_gwish(thetaMat, params);
			 // invG = inv(G);
			 invG = - arma::inv_sympd(hess_gwish(thetaMat, params));
			 thetaNew = theta + stepSize * invG * grad_gwish(thetaMat, params);
			 thetaNewMat = create_psi_mat_cpp(thetaNew, params);
			 psiNew = psi_cpp_mat(thetaNewMat, params);
			 psiCurr = psi_cpp_mat(thetaMat, params);
			 if (-psiNew < -psiCurr) {
				 return(arma::conv_to<arma::vec>::from(theta));
			 }
			 tolCriterion = std::abs(psiNew - psiCurr);
			 theta = thetaNew;
			 thetaMat = thetaNewMat;
			 numSteps++;
		 }
		 /* end newton's method loop */
		 if (numSteps == maxSteps) {
			 Rcpp::Rcout<< "Max # of steps reached in Newton's method." <<
			 	std::endl;
		 } else if (VERBOSE) {
			 Rcpp::Rcout << "Newton's conveged in " << numSteps << " iters" <<
			 	std::endl;
		 }
		 return(arma::conv_to<arma::vec>::from(theta));
} //end calcMode() function


/* ---------------------  approximation functions --------------------------- */

// [[Rcpp::export]]
double approx_integral(u_int K, arma::mat& psi_df, arma::mat& bounds,
	Rcpp::List& params) {

	u_int D           = params["D"];    // dimension of parameter space

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

		arma::mat psi_mat = create_psi_mat_cpp(u_k, params);

		// Rcpp::Rcout<< u_k << std::endl;
		H_k = hess_gwish(psi_mat, params); // 11/9: using general hessian
		H_k_inv = inv(H_k);
		lambda_k = grad_gwish(psi_mat, params); // 11/9: using general gradient
		b_k = H_k * u_k - lambda_k;
		m_k = H_k_inv * b_k;

		/*
		if (k == (K-1)) {
			Rcpp::Rcout<< m_k << std::endl;
		}
		*/


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

			// Rcpp::Rcout << (0.5 * u_k.t() * H_k * u_k).eval()(0,0) << std::endl;
			// Rcpp::Rcout << (0.5 + m_k.t() * H_k * m_k).eval()(0,0) << std::endl;

		// float x =  (m_k.t() * H_k * m_k).eval()(0,0);
		// Rcpp::Rcout << x << std::endl;

	} // end for() over k

	// TODO: find log-sum-exp function in arma
	return lse(log_terms, K);
} // end approx_integral() function


// [[Rcpp::export]]
arma::mat create_psi_mat_cpp(arma::vec u, Rcpp::List& params) {

	u_int p           = params["p"];    // dimension of the graph G
	u_int D           = params["D"];    // dimension of parameter space
	u_int b           = params["b"];    // degrees of freedom
	arma::vec nu_i    = params["nu_i"]; // see p. 329 of Atay (step 2)
	arma::vec b_i     = params["b_i"];  // see p. 329 of Atay (step 2)
	arma::mat P       = params["P"];    // upper cholesky factor of V_n
	arma::mat G       = params["G"];    // graph G


	arma::vec edgeInd = params["edgeInd"];
	//Rcpp::Rcout << "hello" << std::endl;

	/* convert u into the matrix version with free-elements populated */
	// boolean vectorized version of G
	//arma::vec G_bool  = params["FREE_PARAMS_ALL"];
	// arma::uvec ids = find(G_bool); // extract indices of the free elements
	arma::uvec ids =  params["free_index"];
	arma::vec u_prime(p * p, arma::fill::zeros);
	u_prime.elem(ids) = u;
	// Rcpp::Rcout << ids << std::endl;

	arma::mat u_mat(D, D, arma::fill::zeros);
	u_mat = reshape(u_prime, p, p);
	// Rcpp::Rcout << edgeInd << std::endl;

	/* compute the non-free elmts in the matrix using the free elmts */

	// float test = sum(u_mat[i,i:(j-1)] * P[i:(j-1), j]);
	// arma::mat x1 = psi_mat.submat(0, 0, 0, 3);
	//Rcpp::Rcout << x1 << std::endl;
	// Rcpp::Rcout << psi_mat.submat(0, 0, 3, 0) << std::endl;
	// Rcpp::Rcout << arma::dot(x1, psi_mat.submat(0, 0, 3, 0)) << std::endl;

	// u_mat should have all free elements in it
	double x0, tmp1, tmp2;
	for (u_int i = 0; i < p; i++) {
		for (u_int j = i; j < p; j++) {
			if (G(i,j) > 0) {
				continue; // free element, so these already have entries
			}
			if (i == 0) { // first row
				// TODO: double check this calculation
				u_mat(i,j) = -1/P(j,j) * arma::dot(u_mat.submat(i, i, i, j-1),
												   P.submat(i, j, j-1, j));
			} else {
				x0 = -1/P(j,j) * arma::dot(u_mat.submat(i, i, i, j-1),
										   P.submat(i, j, j-1, j));

				arma::vec tmp(i, arma::fill::zeros);
				for (u_int r = 0; r < i; r++) {
					tmp1 = u_mat(r,i) + arma::dot(u_mat.submat(r, r, r, i-1),
											P.submat(r, i, i-1, i)) / P(i,i);
					tmp2 = u_mat(r,j) + arma::dot(u_mat.submat(r, r, r, j-1),
											P.submat(r, j, j-1, j)) / P(j,j);
					tmp(r) = tmp1 * tmp2;
				}

				u_mat(i,j) = x0 - 1 / u_mat(i,i) * arma::sum(tmp);
			}
		} // end inner for() over j
	} // end outer for() over i

	return u_mat;
} // end create_psi_mat_cpp() function


// [[Rcpp::export]]
double psi_cpp(arma::vec& u, Rcpp::List& params) {

	u_int p           = params["p"];    // dimension of the graph G
	u_int D           = params["D"];    // dimension of parameter space
	u_int b           = params["b"];    // degrees of freedom
	arma::vec nu_i    = params["nu_i"]; // see p. 329 of Atay (step 2)
	arma::vec b_i     = params["b_i"];  // see p. 329 of Atay (step 2)
	arma::mat P       = params["P"];    // upper cholesky factor of V_n
	arma::mat G       = params["G"];    // graph G

	arma::mat psi_mat = create_psi_mat_cpp(u, params);

	double psi_u = p * std::log(2);
	for (u_int i = 0; i < p; i++) {
		psi_u += (b + b_i(i) - 1) * std::log(P(i, i)) +
			(b + nu_i(i) - 1) * std::log(psi_mat(i, i)) -
			0.5 * std::pow(psi_mat(i,i), 2);
		for (u_int j = i + 1; j < p; j++) {
			psi_u += -0.5 * std::pow(psi_mat(i,j), 2);
		}
	}

	return -psi_u;
} // end of psi_cpp() function


// [[Rcpp::export]]
double psi_cpp_mat(arma::mat& psi_mat, Rcpp::List& params) {

	u_int p           = params["p"];    // dimension of the graph G
	u_int D           = params["D"];    // dimension of parameter space
	u_int b           = params["b"];    // degrees of freedom
	arma::vec nu_i    = params["nu_i"]; // see p. 329 of Atay (step 2)
	arma::vec b_i     = params["b_i"];  // see p. 329 of Atay (step 2)
	arma::mat P       = params["P"];    // upper cholesky factor of V_n
	arma::mat G       = params["G"];    // graph G

	double psi_u = p * std::log(2);
	for (u_int i = 0; i < p; i++) {
		psi_u += (b + b_i(i) - 1) * std::log(P(i, i)) +
			(b + nu_i(i) - 1) * std::log(psi_mat(i, i)) -
			0.5 * std::pow(psi_mat(i,i), 2);
		for (u_int j = i + 1; j < p; j++) {
			psi_u += -0.5 * std::pow(psi_mat(i,j), 2);
		}
	}
	return -psi_u;
} // end psi_cpp_mat() function


/** -------------------- end of implementation ----------------------------- **/





/** ---------------- old code for diagonal scale matrix -------------------- **/


/**
arma::vec chol2vec(arma::mat& M, u_int D) {

	u_int k = 0;
	u_int D_0 = D * (D + 1) / 2;
	arma::vec u(D_0);
	for (u_int c = 0; c < D; c++) {
		for (u_int r = 0; r <= c; r++) {
			u(k++) = M(r, c);
		}
	}
	return u;
}

// new gradient functions
// [[Rcpp::export]]
arma::vec g(arma::mat& psi, Rcpp::List& params) {

	u_int D           = params["D"];          // dimension of parameter space
	arma::mat G       = params["G"];          // graph G
	// u_int n_nonfree   = params["n_nonfree"];  // number of nonfree elements
	arma::mat ind_mat = params["t_ind"];      // index of the free elements
	// arma::mat vbar    = params["vbar"];       // index of nonfree elements

	u_int d;
	u_int i, j;
	arma::vec grad_vec(D, arma::fill::zeros);
	for (d = 0; d < D; d++) {
		i = ind_mat(d, 0) - 1; // row of free element
		j = ind_mat(d, 1) - 1; // col of free element
		grad_vec(d) = gg(i, j, psi, params);
	}

	return grad_vec;
} // end g() function

// [[Rcpp::export]]
double gg(u_int i, u_int j, arma::mat& psi, Rcpp::List& params) {

	u_int n_nonfree   = params["n_nonfree"];    // number of nonfree elements
	arma::mat vbar    = params["vbar"];         // index of nonfree elements
	u_int b           = params["b"];            // degrees of freedom
	arma::vec nu_i    = params["nu_i"];         //

	u_int a;
	u_int r, s; // row, col index of non-free elements
	double d_ij = 0;
	for (a = 0; a < n_nonfree; a++) {
		r = vbar(a, 0) - 1;
		s = vbar(a, 1) - 1;
		d_ij += psi(r, s) * dpsi(r, s, i, j, psi, params);
	}

	if (i == j) {
		return d_ij - (b + nu_i(i) - 1) / psi(i,i) + psi(i,i);
	} else {
		return d_ij + psi(i,j);
	}

}
*/


/* gradient functions for diagonal scale matrix --------------------------------

arma::vec grad_cpp(arma::vec& u, Rcpp::List& params);
arma::vec grad_cpp_mat(arma::mat& psi_mat, Rcpp::List& params);
double dpsi_cpp(u_int i, u_int j, arma::mat& psi_mat, Rcpp::List& params);
double dpsi_rsij(u_int r, u_int s, u_int i, u_int j,
	arma::mat& psi_mat, arma::mat& G);


// [[Rcpp::export]]
arma::vec grad_cpp(arma::vec& u, Rcpp::List& params) {


	arma::mat G       = params["G"]; // graph G represented as adjacency matrix
	u_int p           = params["p"]; // dimension of the graph G
	// arma::vec edgeInd = params["edgeInd"];
	u_int D           = params["D"]; // dimension of parameter space
	arma::uvec free   = params["free_index"];
	// TODO: implement create_psi_mat() function later; for now, we pass it in
	// arma::mat psi_mat = vec2chol(u, p)

	arma::mat psi_mat = create_psi_mat_cpp(u, params);

	// initialize matrix that can store the gradient elements
	arma::mat gg(p, p, arma::fill::zeros);
	// populate the gradient matrix entry by entry
	for (u_int i = 0; i < p; i++) {
		for (u_int j = i; j < p; j++) {
			if (G(i,j) > 0) {
				gg(i,j) = dpsi_cpp(i, j, psi_mat, params);
			}
		}
	}
	// convert the matrix back into a vector and return only the entries
	// that have a corresponding edge in the graph
	// arma::vec grad_vec = chol2vec(gg, p);
	// arma::uvec ids = find(edgeInd);
	// return grad_vec.elem(ids);
	return gg.elem(free);
}


// [[Rcpp::export]]
arma::vec grad_cpp_mat(arma::mat& psi_mat, Rcpp::List& params) {


  arma::mat G       = params["G"]; // graph G represented as adjacency matrix
  u_int p           = params["p"]; // dimension of the graph G
  // arma::vec edgeInd = params["edgeInd"];
  u_int D           = params["D"]; // dimension of parameter space
  arma::uvec free   = params["free_index"];
  // TODO: implement create_psi_mat() function later; for now, we pass it in
  // arma::mat psi_mat = vec2chol(u, p)

  // arma::mat psi_mat = create_psi_mat_cpp(u, params);

  // initialize matrix that can store the gradient elements
  arma::mat gg(p, p, arma::fill::zeros);
  // populate the gradient matrix entry by entry
  for (u_int i = 0; i < p; i++) {
	for (u_int j = i; j < p; j++) {
	  if (G(i,j) > 0) {
		gg(i,j) = dpsi_cpp(i, j, psi_mat, params);
	  }
	}
  }
  // convert the matrix back into a vector and return only the entries
  // that have a corresponding edge in the graph
  // arma::vec grad_vec = chol2vec(gg, p);
  // arma::uvec ids = find(edgeInd);

  // Rcpp::Rcout << ids << std::endl;
  // return grad_vec.elem(ids);
  return gg.elem(free);
}


// [[Rcpp::export]]
double dpsi_cpp(u_int i, u_int j, arma::mat& psi_mat, Rcpp::List& params) {

	arma::mat G     = params["G"];    // graph G represented as adjacency matrix
	u_int p         = params["p"];    // dimension of the graph G
	u_int b         = params["b"];    // degrees of freedom
	arma::vec nu_i  = params["nu_i"]; //

	double d_ij; // derivative of psi wrt psi_ij (summation over
				// the derivatives wrt the free elements)

	if (G(i, j) == 0) {
		return 0;
	}
	if (i == j) {

		d_ij = 0;

		for (u_int r = 0; r < p; r++) {
			for (u_int s = r; s < p; s++) {
				if (G(r,s) == 0) {
					// if psi_rs == 0, no derivative calculation, skip
					if (psi_mat(r,s) == 0) {
						continue;
					}
					// otherwise: call the derivative function
					d_ij += psi_mat(r,s) * dpsi_rsij(r, s, i, j, psi_mat, G);
				} // end if for checking G[r,s]
			} // end loop over s
		} // end loop over r

		return d_ij - (b + nu_i(i) - 1) / psi_mat(i,i) + psi_mat(i,i);

	} else {

		d_ij = 0;

		for (u_int r = 0; r < p; r++) {
			for (u_int s = r; s < p; s++) {
				if (G(r,s) == 0) {
					if (psi_mat(r,s) == 0) {
						continue;
					}
					d_ij += psi_mat(r,s) * dpsi_rsij(r, s, i, j, psi_mat, G);
					// Rcpp::Rcout << "G[" << r+1 << ", " << s+1 << \
					// 		"] = " << G(r,s) << std::endl;
				}
			} // end loop over s
		} // end loop over r

	} // end if-else

	// only get to this return statement if we go through the else()
	return d_ij + psi_mat(i,j);

} // end dpsi() function



// dpsi_rsij() : compute derivative d psi_rs / d psi_ij
// [[Rcpp::export]]
double dpsi_rsij(u_int r, u_int s, u_int i, u_int j,
	arma::mat& psi_mat, arma::mat& G) {

	// TODO: fill in the rest of the implementation
	if (G(r, s) > 0) {
		if ((r == i) && (s == j)) { // d psi_{ij} / d psi_{ij} = 1
			return 1;
		} else { // d psi_{rs} / d psi_{ij} = 0, since psi_rs is free
			return 0;
		}
	}

	if (i > r)                                { return 0; }
	if ((i == r) && (j > s))                  { return 0; }
	if ((i == r) && (j == s) && G(r, s) > 0)  { return 1; } // redundant check?
	if ((i == r) && (j == s) && G(r, s) == 0) { return 0; } // d wrt to non-free

	if ((r == 0) && (s > r)) { return 0; } // 1st row case -> simplified formula

	if (r > 0) {

		// arma::vec tmp_sum(r - 1);
		arma::vec tmp_sum(r);

		for (u_int k = 0; k < r; k++) {

			if ((psi_mat(k,s) == 0) && (psi_mat(k,r) == 0)) {
				tmp_sum(k) = 0;
				continue;
			} else {

				if (psi_mat(k,s) == 0) {
					tmp_sum(k) = -1/psi_mat(r,r) *
						dpsi_rsij(k, s, i, j, psi_mat, G) * psi_mat(k,r);
				} else if (psi_mat(k,r) == 0) {
					tmp_sum(k) = -1/psi_mat(r,r) *
						dpsi_rsij(k, r, i, j, psi_mat, G) * psi_mat(k,s);
				} else {

					if ((i == j) && (r == i) && (G(r,s) == 0)) {
						tmp_sum(k) = 1/std::pow(psi_mat(r,r), 2) *
						psi_mat(k,r) * psi_mat(k,s) -
							1/psi_mat(r,r) *
							(dpsi_rsij(k, r, i, j, psi_mat, G) * psi_mat(k, s) +
							 dpsi_rsij(k, s, i, j, psi_mat, G) * psi_mat(k, r));
					} else {
						tmp_sum(k) = -1/psi_mat(r,r) * (
						  dpsi_rsij(k, r, i, j, psi_mat, G) * psi_mat(k, s) +
						  dpsi_rsij(k, s, i, j, psi_mat, G) * psi_mat(k,r));
					}
				}
			} // end if-else
		} // end for

		// expression derived from Eq. (99)
		return arma::sum(tmp_sum);

	} else {
		return -999;
	}

} // end dpsi_rsij() function

end of gradient chunk ------------------------------------------------------ **/





/* hessian functions for diagonal scale matrix ---------------------------------


arma::mat hess_cpp(arma::vec& u, Rcpp::List& params);
arma::mat hess_cpp_mat(arma::mat& psi_mat, Rcpp::List& params);
double d2(u_int i, u_int j, u_int k, u_int l,
	arma::mat& psi_mat, Rcpp::List& params);
double d2_rs(u_int r, u_int s, u_int i, u_int j, u_int k, u_int l,
	arma::mat& psi_mat, arma::mat& G);
double d2psi_ii(u_int r, u_int s, u_int i, arma::mat& psi_mat);


// [[Rcpp::export]]
arma::mat hess_cpp_test(arma::vec& u, Rcpp::List& params) {

  u_int D           = params["D"];          // dimension of parameter space
  arma::mat G       = params["G"];          // graph G
  u_int n_nonfree   = params["n_nonfree"];  // # nonfree elements
  arma::mat ind_mat = params["t_ind"];      // index of the free elements
  arma::mat vbar    = params["vbar"];       // index of nonfree elements
  u_int b           = params["b"];          // degrees of freedom
  arma::vec nu_i    = params["nu_i"];       //
  arma::mat psi_mat = create_psi_mat_cpp(u, params);

  arma::mat H(D, D, arma::fill::zeros);

  u_int d, i, j, a, r, c, rr, ss, k, l; // initialize various indices
  double tmp;

  // first populate the diagonal elements of the hessian matrix
  for (d = 0; d < D; d++) {
	// subtract one to account for 0-index in C++
	i = ind_mat(d, 0) - 1; // row loc of d-th free element
	j = ind_mat(d, 1) - 1; // col loc of d-th free element

	if (i == j) { // diagonal free elements
	  tmp = 0;
	  for (a = 0; a < n_nonfree; a++) {

		rr = vbar(a, 0) - 1; // row index of non-free element
		ss = vbar(a, 1) - 1; // col index of non-free element

		tmp += std::pow(dpsi_rsij(rr, ss, i, j, psi_mat, G), 2) +
		  psi_mat(rr,ss) * d2psi_ii(rr, ss, i, psi_mat);
	  } // end of iteration over non-free elements

	  H(d,d) = (b + nu_i(i) - 1) / std::pow(psi_mat(i,i), 2) + 1 + tmp;
	} else {

	  tmp = 0;
	  for (a = 0; a < n_nonfree; a++) {
		rr = vbar(a, 0) - 1; // row index of non-free element
		ss = vbar(a, 1) - 1; // col index of non-free element

		// 11/5/21: previous implementation (line commented out below)
		// did not account for the 2nd order derivative term
		// tmp += std::pow(dpsi_rsij(rr, ss, i, j, psi_mat, G), 2)
		tmp += std::pow(dpsi_rsij(rr, ss, i, j, psi_mat, G), 2) +
		  psi_mat(rr, ss) * d2_rs(rr, ss, i, j, i, j, psi_mat, G);
	  }

	  H(d,d) = 1 + tmp;
	} // end if-else
  } // end for() over d


  // populate the off-diagonal elements of H
  for (r = 0; r < (D-1); r++) { // index should be correct now

	i = ind_mat(r,0) - 1;
	j = ind_mat(r,1) - 1;

	for (c = r + 1; c < D; c++) { // index should be correct

	  k = ind_mat(c, 0) - 1;
	  l = ind_mat(c, 1) - 1;

	  // H(r,c) = d2(i, j, k, l, psi_mat, params);
	  // 11/5/21: testing to account for missing first order partial term
	  H(r,c) = dpsi_rsij(i, j, k, l, psi_mat, G) +
				d2(i, j, k, l, psi_mat, params);
	  H(c,r) = H(r,c); // reflect calculation across diagonal
	} // end inner for() over cols
  } // end outer for() over rows
  return H;
}



// [[Rcpp::export]]
double d2psi_ii(u_int r, u_int s, u_int i, arma::mat& psi_mat) {

	 double out = 0;
	// if (i == r)
	if (i == r) {
	  for (u_int m = 0; m < r; m++) {
		out += psi_mat(m,r) * psi_mat(m,s);
	  }


	} else {
	  return out;
	}


	return(-2 / std::pow(psi_mat(r,r), 3) * out);
}


// [[Rcpp::export]]
arma::mat hess_cpp(arma::vec& u, Rcpp::List& params) {

	u_int D           = params["D"];          // dimension of parameter space
	arma::mat G       = params["G"];          // graph G
	u_int n_nonfree   = params["n_nonfree"];  // # nonfree elements
	arma::mat ind_mat = params["t_ind"];      // index of the free elements
	arma::mat vbar    = params["vbar"];       // index of nonfree elements
	u_int b           = params["b"];          // degrees of freedom
	arma::vec nu_i    = params["nu_i"];       //
	arma::mat psi_mat = create_psi_mat_cpp(u, params);

	arma::mat H(D, D, arma::fill::zeros);

	u_int d, i, j, a, r, c, rr, ss, k, l; // initialize various indices
	double tmp;

	// first populate the diagonal elements of the hessian matrix
	for (d = 0; d < D; d++) {
		// subtract one to account for 0-index in C++
		i = ind_mat(d, 0) - 1; // row loc of d-th free element
		j = ind_mat(d, 1) - 1; // col loc of d-th free element

		if (i == j) {
			tmp = 0;
			for (a = 0; a < n_nonfree; a++) {

				rr = vbar(a, 0) - 1; // row index of non-free element
				ss = vbar(a, 1) - 1; // col index of non-free element

				tmp += std::pow(dpsi_rsij(rr, ss, i, j, psi_mat, G), 2);
			} // end of iteration over non-free elements

			H(d,d) = (b + nu_i(i) - 1) / std::pow(psi_mat(i,i), 2) + 1 + tmp;
		} else {

			tmp = 0;
			for (a = 0; a < n_nonfree; a++) {
				rr = vbar(a, 0) - 1; // row index of non-free element
				ss = vbar(a, 1) - 1; // col index of non-free element

				tmp += std::pow(dpsi_rsij(rr, ss, i, j, psi_mat, G), 2);
			}

			H(d,d) = 1 + tmp;
		} // end if-else
	} // end for() over d


	// populate the off-diagonal elements of H
	for (r = 0; r < (D-1); r++) { // index should be correct now

		i = ind_mat(r,0) - 1;
		j = ind_mat(r,1) - 1;

		for (c = r + 1; c < D; c++) { // index should be correct

			k = ind_mat(c, 0) - 1;
			l = ind_mat(c, 1) - 1;

			H(r,c) = d2(i, j, k, l, psi_mat, params);
			H(c,r) = H(r,c); // reflect calculation across diagonal
		} // end inner for() over cols
	} // end outer for() over rows
	return H;
}

// [[Rcpp::export]]
arma::mat hess_cpp_mat(arma::mat& psi_mat, Rcpp::List& params) {

  u_int D           = params["D"];          // dimension of parameter space
  arma::mat G       = params["G"];          // graph G
  u_int n_nonfree   = params["n_nonfree"];  // # nonfree elements
  arma::mat ind_mat = params["t_ind"];      // index of the free elements
  arma::mat vbar    = params["vbar"];       // index of nonfree elements
  u_int b           = params["b"];          // degrees of freedom
  arma::vec nu_i    = params["nu_i"];       //
  // arma::mat psi_mat = create_psi_mat_cpp(u, params);

  arma::mat H(D, D, arma::fill::zeros);

  u_int d, i, j, a, r, c, rr, ss, k, l; // initialize various indices
  double tmp;

  // first populate the diagonal elements of the hessian matrix
  for (d = 0; d < D; d++) {
	// subtract one to account for 0-index in C++
	i = ind_mat(d, 0) - 1; // row loc of d-th free element
	j = ind_mat(d, 1) - 1; // col loc of d-th free element

	if (i == j) {
	  tmp = 0;
	  for (a = 0; a < n_nonfree; a++) {

		rr = vbar(a, 0) - 1; // row index of non-free element
		ss = vbar(a, 1) - 1; // col index of non-free element

		tmp += std::pow(dpsi_rsij(rr, ss, i, j, psi_mat, G), 2);
	  } // end of iteration over non-free elements

	  H(d,d) = (b + nu_i(i) - 1) / std::pow(psi_mat(i,i), 2) + 1 + tmp;
	} else {

	  tmp = 0;
	  for (a = 0; a < n_nonfree; a++) {
		rr = vbar(a, 0) - 1; // row index of non-free element
		ss = vbar(a, 1) - 1; // col index of non-free element

		tmp += std::pow(dpsi_rsij(rr, ss, i, j, psi_mat, G), 2);
	  }

	  H(d,d) = 1 + tmp;
	} // end if-else
  } // end for() over d


  // populate the off-diagonal elements of H
  for (r = 0; r < (D-1); r++) { // index should be correct now

	i = ind_mat(r,0) - 1;
	j = ind_mat(r,1) - 1;

	for (c = r + 1; c < D; c++) { // index should be correct

	  k = ind_mat(c, 0) - 1;
	  l = ind_mat(c, 1) - 1;

	  H(r,c) = d2(i, j, k, l, psi_mat, params);
	  H(c,r) = H(r,c); // reflect calculation across diagonal
	} // end inner for() over cols
  } // end outer for() over rows
  return H;
}


// [[Rcpp::export]]
double d2(u_int i, u_int j, u_int k, u_int l,
	arma::mat& psi_mat, Rcpp::List& params) {

	u_int n_nonfree   = params["n_nonfree"];  // # nonfree elements
	arma::mat G       = params["G"];          // graph G
	arma::mat vbar    = params["vbar"];       // index of nonfree elements

	arma::vec tmp(n_nonfree, arma::fill::zeros);

	u_int n, r, s;
	for (n = 0; n < n_nonfree; n++) {
		r = vbar(n, 0) - 1; // row index of nonfree element
		s = vbar(n, 1) - 1; // col index of nonfree element
		if (psi_mat(r,s) == 0) { // can be combined partially with step below
			tmp(n) = dpsi_rsij(r, s, k, l, psi_mat, G) *
				dpsi_rsij(r, s, i, j, psi_mat, G);
		} else {
			// product rule: (d psi_rs / d psi_ij d psi_kl)^2 +
	  //               psi_rs * (d^2 psi_rs / d psi_ij ^ 2)
			tmp(n) = dpsi_rsij(r, s, k, l, psi_mat, G) *
				dpsi_rsij(r, s, i, j, psi_mat, G) +
				psi_mat(r,s) * d2_rs(r, s, i, j, k, l, psi_mat, G);
			// tmp(n) = d2_rs(r, s, i, j, k, l, psi_mat, G);
		} // end of if-else

	} // end for() over nonfree elements
	return arma::sum(tmp);
} // end d2() function


// [[Rcpp::export]]
double d2_rs(u_int r, u_int s, u_int i, u_int j, u_int k, u_int l,
	arma::mat& psi_mat, arma::mat& G) {

	 // assumption: we don't have i == k AND j == l, i.e., computing 2nd order
     // derivative for diagonal term in the hessian matrix, since we've already
     // done this previously in a loop dedicated to computing diagonal terms
	 // in the hessian matrix


	// check for early break condtions to save computing
	if (G(r, s) > 0) { return 0; } 			// free element
	if ((r < i) || (r < k)) { return 0; }   // row below
	// same row, col after
    // 11/4/21: condition below is too stringent, r doesn't need to equal BOTH i
    // AND k. Instead, we should check if either is equal and then provide
    // a check for their corresponding columns to see i
	// if ((r == i) && (r == k) && ((s < j) || (s < l))) { return 0; }

    // 11/4/21: revised condition see notes to see explanation:
    // TODO: add in the explanation for this one later
    if (((r == i) && (s < j)) || ((r == k) && (s < l))) { return 0;}


	// general case: recursive call for 2nd order derivative
	// see note below in loop for why vector is size r instead of size (r-1)
	arma::vec tmp(r, arma::fill::zeros);
	u_int m;
	for (m = 0; m < r; m++) {
		// Note on the 0-indexing and upper limit of the loop:
		// Because r represents the *row number*, in the R code, r = 3 for row 3
		// so the loop will run 1:(3-1) -> 2 times, but in the C++ code, we do
		// not need to subtract 1 from the upper limit because row 3 corresponds
		// to r = 2, so the loop will run 0:1 -> 2 times, matching the number
		// of iterations run in the R code


		if ((psi_mat(m, s) == 0) && (psi_mat(m, r) == 0)) {
			tmp(m) = - 1 / psi_mat(r, r) * (
				dpsi_rsij(m, r, i, j, psi_mat, G) *
				dpsi_rsij(m, s, k, l, psi_mat, G) +
				dpsi_rsij(m, r, k, l, psi_mat, G) *
				dpsi_rsij(m, s, i, j, psi_mat, G)
			);
		} else {

			if ((r == i) && (i == j)) { // case: d^2(psi_rs) / (dpsi_rr dpsi_kl)

				if (psi_mat(m, s) == 0) {
					tmp(m) = 1 / std::pow(psi_mat(r, r), 2) *
						(psi_mat(m, r) * dpsi_rsij(m, s, k, l, psi_mat, G)) -
						1 / psi_mat(r, r) * (
							d2_rs(m, s, i, j, k, l, psi_mat, G) * psi_mat(m,r) +
							dpsi_rsij(m, r, i, j, psi_mat, G) *
							dpsi_rsij(m, s, k, l, psi_mat, G) +
							dpsi_rsij(m, r, k, l, psi_mat, G) *
							dpsi_rsij(m, s, i, j, psi_mat, G)
						);

				} else if (psi_mat(m, r) == 0) {
					tmp(m) = 1 / std::pow(psi_mat(r, r), 2) *
						(dpsi_rsij(m, r, k, l, psi_mat, G) * psi_mat(m, s)) -
						1 / psi_mat(r, r) * (
							d2_rs(m, r, i, j, k, l, psi_mat, G) * psi_mat(m,s) +
							dpsi_rsij(m, r, i, j, psi_mat, G) *
							dpsi_rsij(m, s, k, l, psi_mat, G) +
							dpsi_rsij(m, r, k, l, psi_mat, G) *
							dpsi_rsij(m, s, i, j, psi_mat, G)
						);

				} else {
					tmp(m) = 1 / std::pow(psi_mat(r, r), 2) *
						(dpsi_rsij(m, r, k, l, psi_mat, G) * psi_mat(m, s) +
						 psi_mat(m, r) * dpsi_rsij(m, s, k, l, psi_mat, G)) -
						1 / psi_mat(r, r) * (
							d2_rs(m, r, i, j, k, l, psi_mat, G) * psi_mat(m,s) +
							d2_rs(m, s, i, j, k, l, psi_mat, G) * psi_mat(m,r) +
							dpsi_rsij(m, r, i, j, psi_mat, G) *
							dpsi_rsij(m, s, k, l, psi_mat, G) +
							dpsi_rsij(m, r, k, l, psi_mat, G) *
							dpsi_rsij(m, s, i, j, psi_mat, G)
						);
				} //

			} else if ((r == k) && (k == l)) {
				tmp(m) = d2_rs(r, s, k, l, i, j, psi_mat, G);
			} else { // case when r != i
				if (psi_mat(m, s) == 0) {
					tmp(m) = -1 / psi_mat(r, r) * (
						dpsi_rsij(m,r,i,j,psi_mat,G) *
						dpsi_rsij(m,s,k,l,psi_mat,G) +
						dpsi_rsij(m,r,k,l,psi_mat,G) *
						dpsi_rsij(m,s,i,j,psi_mat,G) +
						psi_mat(m,r) * d2_rs(m, s, i, j, k, l, psi_mat, G)
					);

				} else if (psi_mat(m, r) == 0) {
					tmp(m) = - 1 / psi_mat(r, r) * (
						dpsi_rsij(m, r, i, j, psi_mat, G) *
						dpsi_rsij(m, s, k, l, psi_mat, G) +
						dpsi_rsij(m, r, k, l, psi_mat, G) *
						dpsi_rsij(m, s, i, j, psi_mat, G) +
						psi_mat(m, s) * d2_rs(m, r, i, j, k, l, psi_mat, G)
					);
				} else {
					tmp(m) = - 1 / psi_mat(r, r) * (
						dpsi_rsij(m, r, i, j, psi_mat, G) *
						dpsi_rsij(m, s, k, l, psi_mat, G) +
						dpsi_rsij(m, r, k, l, psi_mat, G) *
						dpsi_rsij(m, s, i, j, psi_mat, G) +
						psi_mat(m, s) * d2_rs(m, r, i, j, k, l, psi_mat, G) +
						psi_mat(m, r) * d2_rs(m, s, i, j, k, l, psi_mat, G)
					);
				}

			}
		} // end of main if-else
	} // end for() over m

	return arma::sum(tmp);
} // end d2_rs() function

end of hessian functions ---------------------------------------------------- */



// end gwish.cpp file
