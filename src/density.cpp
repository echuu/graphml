
#include "Tree.h" 
#include "density.h"
#include "Graph.h"
#include "tools.h"      // for lse()
#include "epmgp.h"      // for ep_logz() function 
#include "partition.h"  // for findOptPoints()



#define RCPP_ARMADILLO_RETURN_COLVEC_AS_VECTOR
// [[Rcpp::depends(RcppArmadillo)]]
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS


/* --- functions that are common to the different approximation functions --- */


arma::vec calcMode(arma::mat u_df, Graph* graph) {
	double tol = 1e-8;
	u_int maxSteps = 10;
	bool VERBOSE = false;
	
	// use the MAP as starting point for algorithm
	u_int D = graph->D;
	u_int mapIndex = u_df.col(D).index_min();
	
	arma::vec theta = arma::conv_to<arma::vec>::from(
        u_df.row(mapIndex)).subvec(0, D-1);
	
	u_int numSteps = 0;
	double tolCriterion = 100;
	double stepSize = 1;
	
	arma::mat thetaMat = vec2mat(theta, graph);
	arma::mat thetaNew;
	arma::mat thetaNewMat;
	arma::mat G;
	arma::mat invG;
	double psiNew = 0, psiCurr = 0;
	/* start newton's method loop */
	while ((tolCriterion > tol) && (numSteps < maxSteps)) {
		invG = - arma::inv_sympd(hess_gwish(thetaMat, graph));
		thetaNew = theta + stepSize * invG * grad_gwish(thetaMat, graph);
		thetaNewMat = vec2mat(thetaNew, graph);
		psiNew = psi_cpp_mat(thetaNewMat, graph);
		psiCurr = psi_cpp_mat(thetaMat, graph);
		if (-psiNew < -psiCurr) {
			return(arma::conv_to<arma::vec>::from(theta));
		}
		tolCriterion = std::abs(psiNew - psiCurr);
		theta = thetaNew;
		thetaMat = thetaNewMat;
		numSteps++;
	} /* end newton's method loop */
	if (numSteps == maxSteps) {
		Rcpp::Rcout<< "Max # of steps reached in Newton's method." << std::endl;
	} else if (VERBOSE) {
		Rcpp::Rcout << "Converged in " << numSteps << " iters" << std::endl;
	}	
	return arma::conv_to<arma::vec>::from(theta);	
} // end calcMode() function


double h(u_int i, u_int j, arma::mat& L) {
    // used to compute a fraction quantity in the gradient/hessian functions
	// L is UPPER TRIANGULAR cholesky factor of the INVERSE scale matrix, i.e,
	// D^(-1) = L'L
	return L(i, j) / L(j, j);
} // end h() function (used to be called xi())


arma::mat vec2mat(arma::vec u, Graph* graph) {

	u_int p        = graph->p;    // dimension of the graph G
	u_int D        = graph->D;    // dimension of parameter space
	arma::vec nu_i = graph->nu_i; // see p. 329 of Atay (step 2)
	arma::vec b_i  = graph->b_i;  // see p. 329 of Atay (step 2)
	arma::mat P    = graph->P;    // upper cholesky factor of V_n
	arma::mat G    = arma::conv_to<arma::mat>::from(graph->G);
	arma::vec edgeInd = graph->edgeInd;
	arma::uvec ids =  graph->free_index;
	arma::vec u_prime(p * p, arma::fill::zeros);
	u_prime.elem(ids) = u;
	// Rcpp::Rcout << ids << std::endl;

	arma::mat u_mat(D, D, arma::fill::zeros);
	u_mat = reshape(u_prime, p, p);
	// Rcpp::Rcout << edgeInd << std::endl;

	/* compute the non-free elmts in the matrix using the free elmts */

	// float test = sum(u_mat[i,i:(j-1)] * P[i:(j-1), j]);
	// arma::mat x1 = psi_mat.submat(0, 0, 0, 3);
	// Rcpp::Rcout << x1 << std::endl;
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
} // end vec2mat() function



/* ------------- start functions for: density, gradient, hessian ------------ */ 

double psi_cpp(arma::vec& u, Graph* graph) {

	u_int p           = graph->p;    // dimension of the graph G
	u_int b           = graph->b;    // degrees of freedom
	arma::vec nu_i    = graph->nu_i; // see p. 329 of Atay (step 2)
	arma::vec b_i     = graph->b_i;  // see p. 329 of Atay (step 2)
	arma::mat P       = graph->P;    // upper cholesky factor of V_n

	arma::mat psi_mat = vec2mat(u, graph);

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


double psi_cpp_mat(arma::mat& psi_mat, Graph* graph) {

	u_int p           = graph->p;    // dimension of the graph G
	u_int b           = graph->b;    // degrees of freedom
	arma::vec nu_i    = graph->nu_i; // see p. 329 of Atay (step 2)
	arma::vec b_i     = graph->b_i;  // see p. 329 of Atay (step 2)
	arma::mat P       = graph->P;    // upper cholesky factor of V_n

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

/** ------------------- start gradient functions --------------------------- **/

arma::vec grad_gwish(arma::mat& psi_mat, Graph* graph) {
	arma::mat G = arma::conv_to<arma::mat>::from(graph->G);
	u_int p = graph->p; // dimension of the graph G
	arma::uvec free = graph->free_index;
	// initialize matrix that can store the gradient elements
	arma::mat gg(p, p, arma::fill::zeros);
	// populate the gradient matrix entry by entry
	for (u_int i = 0; i < p; i++) {
		for (u_int j = i; j < p; j++) {
			if (G(i,j) > 0) {
				gg(i,j) = dpsi_ij(i, j, psi_mat, graph);
			}
		}
	}
	return gg.elem(free);
} // end grad_gwish() function


double dpsi_ij(u_int i, u_int j, arma::mat& psi_mat, Graph* graph) {

	arma::mat G = arma::conv_to<arma::mat>::from(graph->G);
	u_int p = graph->p;  // dimension of the graph G
	u_int b = graph->b;  // degrees of freedom
	arma::vec nu_i = graph->nu_i; 

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
					d_ij += psi_mat(r,s) * dpsi(r, s, i, j, psi_mat, graph);
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
					d_ij += psi_mat(r,s) * dpsi(r, s, i, j, psi_mat, graph);
					// Rcpp::Rcout << "G[" << r+1 << ", " << s+1 <<
					//      "] = " << G(r,s) << std::endl;
				}
			} // end loop over s
		} // end loop over r
	} // end if-else
	// only get to this return statement if we go through the else()
	return d_ij + psi_mat(i,j);
} // end dpsi_ij() function


double dpsi(u_int r, u_int s, u_int i, u_int j, arma::mat& psi, Graph* graph) {

	arma::mat G = arma::conv_to<arma::mat>::from(graph->G);
	arma::mat L = graph->P;

	// Rcpp::Rcout << "beginning"  << std::endl;
	// Rcpp::Rcout << "G(" << r << ", " << s << ") = " << G(r,s) << std::endl;
	// Rcpp::Rcout << "r =  " << r << ", s =  " << s << ", i =  " << i <<
	//    ", j =  " << j << std::endl;

	if (G(r, s) > 0) {
		if ((r == i) && (s == j)) { // d psi_{ij} / d psi_{ij} = 1
			return 1;
		} else { // d psi_{rs} / d psi_{ij} = 0, since psi_rs is free
			return 0;
		}
	}
	if (i > r)                                { return 0; }
	if ((i == r) && (j > s))                  { return 0; }
	/* 11/5: fairly certain we don't need the following 2 checks */
	if ((i == r) && (j == s) && G(r, s) > 0)  { return 1; } // redundant check?
	if ((i == r) && (j == s) && G(r, s) == 0) { return 0; } // d wrt to non-free

	u_int k, l;
	double x;
	// 1st row case: simplified formulation of free elements
	if (r == 0) {
		// don't need to check (s > r) because that case is already flagged by
		// the initial check in this function: if (G(r,s) > 0)
		x = 0;
		for (k = 0; k < s; k++) {
			x += dpsi(r, k, i, j, psi, graph) * h(k, s, L); // correct G
		}
		return -x;
	} // end row 1 case

	bool DWRT_SAME_ROW_DIAG = ((i == j) && (r == i) && (G(r, s) == 0));
	// bool DWRT_SAME_ROW_DIAG = ((i == j) && (r == i));
	double s0 = 0, s10 = 0, s11 = 0;
	double s12 = 0, s13 = 0, s110 = 0, s111 = 0, s120 = 0, s121 = 0; // i != j
	double out; // store the result of the calculation
	arma::vec s1(r); // store each term in the summation

	// TODO: THIS CHUNK IN PROGRESS
	if (DWRT_SAME_ROW_DIAG) { // dpsi_rs / dpsi_rr
		// Rcpp::Rcout << "HERE"  << std::endl;
		for (k = r; k < s; k++) {
			s0 += h(k, s, L) * dpsi(r, k, i, j, psi, graph);
		}
		for (k = 0; k < r; k++) {
			for (l = k; l < s; l++) {
				s10 += psi(k,l) * h(l, s, L);
			}
			for (l = k; l < r; l++) {
				s11 += psi(k,l) * h(l, r, L);
			}
			s1(k) = psi(k,r) * psi(k,s) +
				psi(k,r) * s10 + psi(k,s) * s11 + s10 * s11;
		} // end inner for()

		out = -s0 + 1 / std::pow(psi(r,r), 2) * arma::sum(s1);

	} else { // dpsi_rs / dpsi_ij
		// general case when derivative is wrt general (i, j), i > 0
		for (k = r; k < s; k++) {
			s0 += h(k, s, L) * dpsi(r, k, i, j, psi, graph);
		}
		/* calculation of s1, inner summation from 1:(r-1)
		   s1 = s10 + s11 + s12 + s13, where each component is one of the four
		   terms in the summation. s13 is consisted of 4 separate summations,
		   for which we use 2 for loops to compute.
		*/
		for (k = 0; k < r; k++) {

			// compute the intermediate summations:
			for (l = k; l < s; l++) {
				s110 += psi(k, l) * h(l, s, L);
				// note: s111 is just s110 w/ derivative wrt (ij) applied to
				// the first term
				s111 += dpsi(k, l, i, j, psi, graph) * h(l, s, L);
			}
			for (l = k; l < r; l++) {
				s120 += psi(k, l) * h(l, r, L);
				// note: s121 is just s120 w/ derivative wrt (ij) applied to
				// the first term
				s121 += dpsi(k, l, i, j, psi, graph) * h(l, r, L);
			}

			s10 = psi(k,s) * dpsi(k, r, i, j, psi, graph) +
				psi(k,r) * dpsi(k, s, i, j, psi, graph);
			s11 = dpsi(k, r, i, j, psi, graph) * s110 + psi(k,r) * s111;
			s12 = dpsi(k, s, i, j, psi, graph) * s120 + psi(k,s) * s121;
			s13 = s121 * s110 + s120 * s111;

			s1(k) = s10 + s11 + s12 + s13;
		} // end of for loop computing EACH TERM of s1 summation from 0:(r-1)
		out = -s0 - 1 / psi(r,r) * arma::sum(s1);
	} // end if-else()
	return out;
} // end dpsi() function


/** -------------------- end gradient functions ---------------------------- **/

/** -------------------- start hessian functions --------------------------- **/

arma::mat hess_gwish(arma::mat& psi_mat, Graph* graph) {

	u_int D           = graph->D;          // dimension of parameter space
	arma::mat G       = arma::conv_to<arma::mat>::from(graph->G);
	u_int n_nonfree   = graph->n_nonfree;  // number of nonfree elements
	arma::mat ind_mat = graph->t_ind;      // index of the free elements
	arma::mat vbar    = graph->vbar;       // index of nonfree elements
	u_int b           = graph->b;          // degrees of freedom
	arma::vec nu_i    = graph->nu_i;       // see Atay paper for definition

	arma::mat H(D, D, arma::fill::zeros);     // (D x D) hessian matrix

	u_int d, i, j, a, r, c, rr, ss, k, l; // initialize various indices
	double x; // store intermediate calculations

          // first populate the diagonal elements of the hessian matrix
	for (d = 0; d < D; d++) {
        // subtract one to account for 0-index in C++
        i = ind_mat(d, 0) - 1; // row loc of d-th free element
        j = ind_mat(d, 1) - 1; // col loc of d-th free element

        if (i == j) { // diagonal free elements
        	x = 0;
        	for (a = 0; a < n_nonfree; a++) {

            rr = vbar(a, 0) - 1; // row index of non-free element
            ss = vbar(a, 1) - 1; // col index of non-free element

            /* the call to d2psi_ii() should be replaced by a more general
               call to d2psi(), as defined below, since this case should be
               captured in the 2nd order mixed partial calculation, but we can
               do this part later.. just set up the architecture for now */
            x += std::pow(dpsi(rr, ss, i, j, psi_mat, graph), 2) +
            psi_mat(rr,ss) * d2psi(rr, ss, i, j, i, j, psi_mat, graph);

        } // end of iteration over non-free elements

        H(d,d) = (b + nu_i(i) - 1) / std::pow(psi_mat(i,i), 2) + 1 + x;

      	} else {
      		/* note: the following for loop over the non-free elements
      		   and the following calculation with the calculations of
      		   (1) the product of two first order derivatives
			   and (2) the product of psi(r,s) and the second order mixed
			   partial of psi(r,s) wrt psi(ij) and psi(ij)
			   is IDENTICAL to the general calculation of d2psi_ijkl in the
			   function below, but we can merge these two calculations later
			   by simply (?) deleteing or calling d2psi_ijkl() from inside
			   this else statement
			   only difference from the chunk below is that there is an
			   additional first order term, which reduces to 1 
			   since dpsi_ij / dpsi_ij = 1
			   and we see this in the H(d,d) = 1 + x:
			   we can fix this later
      		*/
	      	x = 0;
	      	for (a = 0; a < n_nonfree; a++) {
	            rr = vbar(a, 0) - 1; // row index of non-free element
	            ss = vbar(a, 1) - 1; // col index of non-free element

	            // 11/5/21: previous implementation (line commented out below)
	            // did not account for the 2nd order derivative term
	            // tmp += std::pow(dpsi_rsij(rr, ss, i, j, psi_mat, G), 2)
	            x += std::pow(dpsi(rr, ss, i, j, psi_mat, graph), 2) +
	            	psi_mat(rr, ss) * 
					d2psi(rr, ss, i, j, i, j, psi_mat, graph);
	        }

        	H(d,d) = 1 + x;
    	} // end if-else
    } // end for() over d

    // Rcpp::Rcout << "popuated diagonal elements" << std::endl;
    // return H;

	// populate off-diagonal elements of the hessian
	for (r = 0; r < D; r++) {
		i = ind_mat(r, 0) - 1; // row loc of d-th free element
	    j = ind_mat(r, 1) - 1; // col loc of d-th free element
	    for (c = r + 1; c < D; c++) { // index should be correct
	    	k = ind_mat(c, 0) - 1;
	  		l = ind_mat(c, 1) - 1;

	  		H(r,c) = dpsi(i, j, k, l, psi_mat, graph) +
				d2psi_ijkl(i, j, k, l, psi_mat, graph);
	    	H(c, r) = H(r, c); // reflect the value over the diagonal
	    } // end inner for() over upper triangular columns
	} // end for() population off-diagonal elements of the hessian


	return H;
} // end hess_gwish() function


/* d2psi_ijkl():
	this function populated each entry in the hessian matrix. It computes
	the summation over vbar (set of NONFREE elements), where each term in the
	summation consists of (1) the product of two first order derivatives
	and (2) the product of psi(r,s) and the second order mixed partial of
	psi(r,s) wrt psi(ij) and psi(kl)
*/
double d2psi_ijkl(u_int i, u_int j, u_int k, u_int l, arma::mat& psi, 
    Graph* graph) {

	u_int n_nonfree   = graph->n_nonfree;  // # nonfree elements
	arma::mat vbar    = graph->vbar;       // index of nonfree elements
	arma::vec sum_vbar(n_nonfree, arma::fill::zeros); // store summation terms
	u_int n, r, s;
	for (n = 0; n < n_nonfree; n++) {
		r = vbar(n, 0) - 1; // row index of nonfree element
		s = vbar(n, 1) - 1; // col index of nonfree element

		sum_vbar(n) = dpsi(r, s, k, l, psi, graph) *
			dpsi(r, s, i, j, psi, graph) +
			psi(r, s) * d2psi(r, s, i, j, k, l, psi, graph);
	} // end for() over the nonfree elements
	return arma::sum(sum_vbar);
} // end d2psi_ijkl() function


double d2psi(u_int r, u_int s, u_int i, u_int j, u_int k, u_int l,
	arma::mat& psi, Graph* graph) {

	arma::mat G       = arma::conv_to<arma::mat>::from(graph->G);
	arma::mat L       = graph->P;       // UPPER cholesky factor

	/* check terminating conditions; these should break us out of recursion */
	if (G(r, s) > 0) { return 0; } // taking 2nd deriv of free element -> 0
	bool INVALID_ORDER = (
		((r < i) || (r < k))  ||     // numerator comes before denom
		((r == i) && (s < j)) ||     // row match, but 1st order col comes after
		((r == k) && (s < l))        // row match, but 2nd order col comes after
	);
	if (INVALID_ORDER) { return 0; }
	// end of terminating condition checking -----------------------------------

	if ((r == k) && (k == l) && (r != i)) {
		// this should prevent any premature accumulation of x0 (the first
		// summation term). this should also prevent the same check in the for
		// loop to ever hit because we check this immediately
		return d2psi(r, s, k, l, i, j, psi, graph);
	}

	/* compute 2nd order derivative for the general case, i.e,
	   	dpsi(r,s) / ( dpsi(i,j) dpsi(k, l) )

		case 0: dpsi(r,s) / dpsi(r,r) dpsi(r,r)
			- confusing at fist why we need this because it for the hessian
			  we never directly take 2nd order derivative wrt to diagonal, but
			  in the recursive definition of psi(r,s), there is a 1/psi(r,r)
			  term, so when we take 2nd order derivative of J wrt to psi(r,r),
			  psi(r,r), we end up needing to evaluate case 1
		case 1: dpsi(r,s) / dpsi(r,r) dpsi(r,r
		case 2: dpsi(r,s) / dpsi(i,j) dpsi(r,r)
		case 3: dpsi(r,s) / dpsi(i,j) dpsi(k,l)
	*/
	u_int m, n;

	// compute the first summation in the recurisve representation of psi_rs
	double x0 = 0;
	for (m = r; m < s; m++) {
		// Rcpp::Rcout << "(r,s) = (" << r << "," << s << ")" << std::endl;
		x0 += h(m, s, L) * d2psi(r, m, i, j, k, l, psi, graph);
	}
	// Rcpp::Rcout << "x0 = " << x0 << std::endl;

	arma::vec d2_sum_r(r, arma::fill::zeros);
	// compute each term (each term has 4 components) of the summation
	double xi_ns, xi_nr;
	double x1 = 0, x2 = 0, x3 = 0, x4 = 0;
	for (m = 0; m < r; m++) {
		if ((r == i) && (i == j) && (i == k) && (k == l)) {
			// Rcpp::Rcout << "shouldn't be hitting this condition." << std::endl;
			// case 0: dpsi(r,s) / dpsi(r,r) dpsi(r,r)
			double x2_0 = 0, x3_0 = 0;
			for (n = m; n < s; n++) {
				x2_0 += psi(m, n) * h(n, s, L);
			}
			for (n = m; n < r; n++) {
				x3_0 += psi(m, n) * h(n, r, L);
			}
			x1 = psi(m, r) * psi(m, s);
			x2 = psi(m, r) * x2_0;
			x3 = psi(m, s) * x3_0;
			x4 = x2_0 * x3_0;
			d2_sum_r(m) = -2/std::pow(psi(r,r), 3) * (x1 + x2 + x3 + x4);
		} else if ((r == i) && (i == j)) {
			// case 1: d^2(psi_rs) / (dpsi_rr dpsi_kl)
			/*
			Rcpp::Rcout << "case 1"<< std::endl;
			Rcpp::Rcout << "(r,s) = (" << r << "," << s << "), " <<
				"(i,j) = (" << i << "," << j << "), " <<
				"(k,l) = (" << k << "," << l << ")" << std::endl;
			*/
			x1 = psi(m, s) * dpsi(m, r, k, l, psi, graph) +
				 psi(m, r) * dpsi(m, s, k, l, psi, graph);
			double x2_0 = 0, x2_1 = 0;
			for (n = m; n < s; n++) {
				xi_ns = h(n, s, L);
				x2_0 += xi_ns * psi(m, n);
				x2_1 += xi_ns * dpsi(m, n, k, l, psi, graph);
			}
			x2 = dpsi(m, r, k, l, psi, graph) * x2_0 + psi(m, r) * x2_1;

			double x3_0 = 0, x3_1 = 0;
			for (n = m; n < r; n++) {
				xi_nr = h(n, r, L);
				x3_0 += xi_nr * psi(m, n);
				x3_1 += xi_nr * dpsi(m, n, k, l, psi, graph);
			}
			x3 = dpsi(m, s, k, l, psi, graph) * x3_0 + psi(m, s) * x3_1;

			// double x4_0 = 0, x4_1 = 0, x4_2 = 0, x4_3 = 0;
			x4 = x3_1 * x2_0 + x3_0 * x2_1;
			d2_sum_r(m) = 1/std::pow(psi(r,r), 2) * (x1 + x2 + x3 + x4);
		} else if ((r == k) && (k == l)) {
			// case 2: d^2(psi_rs) / (dpsi_ij dpsi_rr)
			// this case should no longer hit because we check this in the very
			// beginning, BEFORE any recursion begins
			d2_sum_r(m) = d2psi(r, s, k, l, i, j, psi, graph);
		} else { // case 3: dpsi(r,s) / dpsi(i,j) dpsi(k,l)
			// double x1 = 0, x2 = 0, x3 = 0, x4 = 0;
			x1 = psi(m, s) * d2psi(m, r, i, j, k, l, psi, graph) +
				 dpsi(m, r, i, j, psi, graph) * dpsi(m, s, k, l, psi, graph) +
				 psi(m, r) * d2psi(m, s, i, j, k, l, psi, graph) +
				 dpsi(m, r, k, l, psi, graph) * dpsi(m, s, i, j, psi, graph);

			/* compute x2 */
			double x2_0 = 0, x2_1 = 0, x2_2 = 0, x2_3 = 0;
			for (n = m; n < s; n++) { // check: ssummation over s
				xi_ns = h(n, s, L);
				x2_0 += xi_ns * psi(m, n);
				x2_1 += xi_ns * dpsi(m, n, k, l, psi, graph);
				x2_2 += xi_ns * dpsi(m, n, i, j, psi, graph);
				x2_3 += xi_ns * d2psi(m, n, i, j, k, l, psi, graph);
			}
			if (x2_1 == 0) {
				x2 = d2psi(m, r, i, j, k, l, psi, graph) * x2_0 +
			     dpsi(m, r, k, l, psi, graph) * x2_2 +
			     psi(m, r) * x2_3;
			} else if (x2_2 == 0) {
				x2 = d2psi(m, r, i, j, k, l, psi, graph) * x2_0 +
			     dpsi(m, r, i, j, psi, graph) * x2_1 +
			     psi(m, r) * x2_3;
			} else {
				x2 = d2psi(m, r, i, j, k, l, psi, graph) * x2_0 +
			     dpsi(m, r, i, j, psi, graph) * x2_1 +
			     dpsi(m, r, k, l, psi, graph) * x2_2 +
			     psi(m, r) * x2_3;
			}
			/* compute x3 */
			double x3_0 = 0, x3_1 = 0, x3_2 = 0, x3_3 = 0;
			for (n = m; n < r; n++) { // check: summation over r
				xi_nr = h(n, r, L);
				x3_0 += xi_nr * psi(m, n);
				x3_1 += xi_nr * dpsi(m, n, k, l, psi, graph);
				x3_2 += xi_nr * dpsi(m, n, i, j, psi, graph);
				x3_3 += xi_nr * d2psi(m, n, i, j, k, l, psi, graph);
			}
			if (x3_1 == 0) {
				x3 = d2psi(m, s, i, j, k, l, psi, graph)  * x3_0 +
			     dpsi(m, s, k, l, psi, graph) * x3_2 +
			     psi(m, s) * x3_3;
			} else if (x3_2 == 0) {
				x3 = d2psi(m, s, i, j, k, l, psi, graph)  * x3_0 +
			     dpsi(m, s, i, j, psi, graph) * x3_1 +
			     psi(m, s) * x3_3;
			} else {
				x3 = d2psi(m, s, i, j, k, l, psi, graph)  * x3_0 +
			     dpsi(m, s, i, j, psi, graph) * x3_1 +
			     dpsi(m, s, k, l, psi, graph) * x3_2 +
			     psi(m, s) * x3_3;
			}
			/* compute x4 */
			double x4_00 = 0, x4_01 = 0, x4_10 = 0, x4_11 = 0, x4_20 = 0,
				x4_21 = 0, x4_30 = 0, x4_31 = 0;
			x4_00 = x3_3; x4_10 = x3_2; x4_20 = x3_1; x4_30 = x3_0;
			x4_01 = x2_0; x4_11 = x2_1; x4_21 = x2_2; x4_31 = x2_3;
			x4 = x4_00 * x4_01 + x4_10 * x4_11 + x4_20 * x4_21 + x4_30 * x4_31;
			d2_sum_r(m) = -1/psi(r,r) * (
				x1 + x2 + x3 + x4
			);
		}
	} // end for() over m, computing each term of the summation

	return - x0 + arma::sum(d2_sum_r);
} // end d2psi() function


// end density.cpp file
