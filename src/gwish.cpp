
#include "gwish.h"
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

// recursive function to determine decision rules
void dfs(Node* node, unsigned int& k, std::vector<Interval*> intervalStack,
    arma::mat& partition, arma::mat& supp, 
    std::unordered_map<u_int, arma::uvec>& leafRowMap) {

    /*  k:          current partition set that we are updating
        decisions:  stack that holdes Interval objects
        partition:  nLeaves x (2 * nFeats) matrix, storing intervals row-wise
    */

    if (node->isLeaf) {
        /*
            extract the interval's feature
            replace the current interval values for that feature with the 
            updated interval values: lb = max(curr, new), ub = min(curr, new);
            this shrinks the interval to accommodate a tighter bound as we
            go deeper into the tree
        */
        for (const auto interval : intervalStack) {
            double lb = interval->lb;
            double ub = interval->ub; 
            unsigned int col = interval->feature; // features are 1-indexed

            // need to take max, min to accommodate for shrinking intervals,
            // further restrictions on the same feature as we go down the tree
            // partition(k, 2*col) = std::max(partition(k, 2*col), lb);
            // partition(k, 2*col+1) = std::min(partition(k, 2*col+1), ub);

            partition(2*col, k) = std::max(partition(2*col, k), lb);
            partition(2*col+1, k) = std::min(partition(2*col+1, k), ub);
        } // end of for() updating the k-th leaf

        // add the rows corresponding to this leaf into the hashmap
        // this is used in the hybrid algo to find candidate/representative 
        // points in each partition (leaf) set
        // NOTE: since we are doing this in the same order as storing the
        // intervals (above), we ensure that the k-th partition set
        // corresponds to the leafRows that we store in the map below
        leafRowMap[k] = node->getLeafRows();

        // Rcpp::Rcout << "row " << k << " : leaf value = " << 
        // node->getLeafVal() << std::endl;
        k++; // move to next leaf node / interval to update
        return;
    } // end of terminating condition

    /*  keep going down the tree and pushing the rules onto the stack until we
        reach a leaf node, then we use all elements in the stack to construct
        the interval. after constructing the interval, pop the leaf node off 
        so we can continue down another path to find the next leaf node
    */ 

    // extract the rule for the curr node -> this will give left and ride nodes
    // features are stored in the Node object as 1-index because the data 
    // looks like [y|X], so features start from column 1 rather than column 0
    // but when creating subsequent maps and matrices, we still want the 
    // first feature to be in the 0-th (first) column (element in map)
    unsigned int feature = node->getFeature() - 1;
    double lb = supp(feature, 0);
    double ub = supp(feature, 1);
    double threshVal = node->getThreshold();  // threshold value for node
    // construct interval for the left node
    Interval* leftInterval = new Interval(lb, threshVal, feature);
    // construct interval for the right node
    Interval* rightInterval = new Interval(threshVal, ub, feature);

    // travel down left node
    intervalStack.push_back(leftInterval);
    dfs(node->left, k, intervalStack, partition, supp, leafRowMap);
    delete intervalStack.back();
    intervalStack.pop_back();

    // travel down right node
    intervalStack.push_back(rightInterval);
    dfs(node->right, k, intervalStack, partition, supp, leafRowMap);
    delete intervalStack.back();
    intervalStack.pop_back();

} // end dfs() function


// testing time complexity of cart algorithm 

// [[Rcpp::export]]
arma::mat timeTreeAndPartition(arma::mat data) {

    Tree* tree = new Tree(data); // this will create the ENTIRE regression tree
    // tree->root will give the root node of the tree
    // double rootThresh = tree->root->getThreshold();
    // Rcpp::Rcout << "Tree has " << tree->getLeaves() << " leaves" << std::endl;

    unsigned int nLeaves = tree->getLeaves();
    unsigned int d = tree->getNumFeats();
    unsigned int n = tree->getNumRows();  
    unsigned int k = 0;

    arma::uvec r = arma::conv_to<arma::uvec>::from(arma::linspace(0, n-1, n));
    arma::uvec c = arma::conv_to<arma::uvec>::from(arma::linspace(1, d, d));
    arma::mat  X = data.submat(r, c);
    arma::mat supp = support(X, d); // extract support
    arma::mat partition = createDefaultPartition(supp, d, nLeaves);

    std::vector<Interval*> intervalStack; 
    std::unordered_map<u_int, arma::uvec> leafRowMap;
    dfs(tree->root, k, intervalStack, partition, supp, leafRowMap);
    /* after the above call to dfs(), the leafRowmap is fully populated and 
       can be used by the rest of the hybrid algorithm */
    delete(tree);
    return partition;
}

// [[Rcpp::export]]
double timeTree(arma::mat data) {

    Tree* tree = new Tree(data); // this will create the ENTIRE regression tree
    // tree->root will give the root node of the tree
    // double rootThresh = tree->root->getThreshold();
    Rcpp::Rcout << "Tree has " << tree->getLeaves() << " leaves" << std::endl;
    // unsigned int nLeaves = tree->getLeaves();
    // unsigned int d = tree->getNumFeats();
    // unsigned int n = tree->getNumRows();  
    // unsigned int k = 0;

    // arma::uvec r = arma::conv_to<arma::uvec>::from(arma::linspace(0, n-1, n));
    // arma::uvec c = arma::conv_to<arma::uvec>::from(arma::linspace(1, d, d));
    // arma::mat  X = data.submat(r, c);
    // arma::mat supp = support(X, d); // extract support
    // arma::mat partition = createDefaultPartition(supp, d, nLeaves);

    // std::vector<Interval*> intervalStack; 
    // std::unordered_map<u_int, arma::uvec> leafRowMap;
    // dfs(tree->root, k, intervalStack, partition, supp, leafRowMap);
    /* after the above call to dfs(), the leafRowmap is fully populated and 
       can be used by the rest of the hybrid algorithm */
    delete(tree);
    return 0;
}

arma::mat timePartition(Tree* tree, arma::mat data) {
	unsigned int nLeaves = tree->getLeaves();
    unsigned int d = tree->getNumFeats();
    unsigned int n = tree->getNumRows();  
    unsigned int k = 0;

    arma::uvec r = arma::conv_to<arma::uvec>::from(arma::linspace(0, n-1, n));
    arma::uvec c = arma::conv_to<arma::uvec>::from(arma::linspace(1, d, d));
    arma::mat  X = data.submat(r, c);
    arma::mat supp = support(X, d); // extract support
    arma::mat partition = createDefaultPartition(supp, d, nLeaves);

    std::vector<Interval*> intervalStack; 
    std::unordered_map<u_int, arma::uvec> leafRowMap;
    dfs(tree->root, k, intervalStack, partition, supp, leafRowMap);

	return partition;
}



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

	// go into here and figure how to use the ROWS of each partition set's 
	// points instead of finding the rows' locations that are equal to
	// the current leaf node / location that we're working on
	std::unordered_map<int, arma::vec> candidates = findAllCandidatePoints(
		data, locs, uStar, D
	);

	// std::unordered_map<int, arma::vec> boundMap = partitionMap;
	 return approxZ(params, leafId, candidates, partitionMap, k);
	 
} // end approxZ() function


/* new implementation starts here ------------------------------------------- */

// [[Rcpp::export]]
double hyb(arma::umat G, u_int b, arma::mat V, u_int J) {

    // Rcpp::Rcout << p << " x " << p << " graph" << std::endl;
    // initialize graph object
    Rcpp::List obj = init_graph(G, b, V);
    // Rcpp::Rcout << "graph initialized" << std::endl;
    // generate J samples from gwishart
    arma::mat samps = rgw(J, obj);
    // Rcpp::Rcout << "obtained samples" << std::endl;
    // evalute the samples using the negative log posterior (psi in last column)

	// TODO: fix evalPsi so that we can use the z = [y | X] instead of [X | y]
    arma::mat samps_psi = evalPsi(samps, obj);
    // Rcpp::Rcout << "evaluated samples" << std::endl;
    // calculate global mode
    arma::vec u_star = calcMode(samps_psi, obj);
    // Rcpp::Rcout << "computed mode" << std::endl;

	// format the data so that it looks like z = [y | X]
	arma::mat z = arma::join_rows( samps_psi.col(samps_psi.n_cols - 1), samps );
	// Rcpp::Rcout << z << std::endl;


    // compute the final approximation
    return approx_v2(z, // need to do some re-arrangement to put response in col 1
                     u_star,
                     samps_psi,
                     obj);
} // end generalApprox() function



double approx_v2(arma::mat z,
				 arma::vec uStar,
				 arma::mat data,
				 Rcpp::List& params) {

	u_int D = params["D"];

	// use new implementation of tree, which returns partition as one of the
	// the fitted tree's member variables
	Tree* tree = new Tree(z, 1);
	std::unordered_map<u_int, arma::vec>* pmap = tree->getPartition();
	std::unordered_map<u_int, arma::uvec>* leafRowMap = tree->getLeafRowMap();
	unsigned int nLeaves = tree->getLeaves();
    unsigned int d = tree->getNumFeats();

	// -------------------------------------------------------------------------

	// go into here and figure how to use the ROWS of each partition set's 
	// points instead of finding the rows' locations that are equal to
	// the current leaf node / location that we're working on
	std::unordered_map<u_int, arma::vec> candidates = findOptPoints(
		data, *leafRowMap, nLeaves, uStar, D
	);

	// std::unordered_map<int, arma::vec> boundMap = partitionMap;
	 return approx_helper(params, candidates, *pmap, nLeaves);
	 
} // end approxZ() function


double approx_helper(Rcpp::List& params, 
					 std::unordered_map<u_int, arma::vec> candidates, 
					 std::unordered_map<u_int, arma::vec> bounds, 
					 u_int nLeaves) {

	u_int D = params["D"];    // dimension of parameter space
	arma::vec log_terms(nLeaves, arma::fill::zeros);
	arma::vec G_k(nLeaves, arma::fill::zeros);
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
	for (u_int k = 0; k < nLeaves; k++) {

		// leaf_k = leaf(k);
		candidate_k = candidates[k];
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

		lb = bounds[k].elem(arma::conv_to<arma::uvec>::from(
			arma::linspace(0, 2 * D - 2, D)));
		ub = bounds[k].elem(arma::conv_to<arma::uvec>::from(
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

	return lse(log_terms, nLeaves);

} // end approx_helper() function


/* new implementation ends -------------------------------------------------- */

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
