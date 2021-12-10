// gwishDensity.cpp

#include "gwishDensity.h"
#include "tools.h"
#define RCPP_ARMADILLO_RETURN_COLVEC_AS_VECTOR
// [[Rcpp::depends(RcppArmadillo)]]
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS


/**** update gradient, hessian to accommodate non-diagonal scale matrices *****/

/** ------------------- start gradient functions --------------------------- **/

// [[Rcpp::export]]
arma::vec grad_gwish(arma::mat& psi_mat, Rcpp::List& params) {

	arma::mat G       = params["G"]; // graph G represented as adjacency matrix
	u_int p           = params["p"]; // dimension of the graph G
	// arma::vec edgeInd = params["edgeInd"];
	arma::uvec free   = params["free_index"];
	u_int D           = params["D"]; // dimension of parameter space
	// TODO: implement create_psi_mat() function later; for now, we pass it in
	// arma::mat psi_mat = vec2chol(u, p)
	// arma::mat psi_mat = create_psi_mat_cpp(u, params);

	// initialize matrix that can store the gradient elements
	arma::mat gg(p, p, arma::fill::zeros);
	// populate the gradient matrix entry by entry
	for (u_int i = 0; i < p; i++) {
		for (u_int j = i; j < p; j++) {
			if (G(i,j) > 0) {
				gg(i,j) = dpsi_ij(i, j, psi_mat, params);
			}
		}
	}
	// convert the matrix back into a vector and return only the entries
	// that have a corresponding edge in the graph
	// arma::vec grad_vec = chol2vec(gg, p);
	// arma::uvec ids = find(edgeInd);
	// return grad_vec.elem(ids);
	return gg.elem(free);
} // end grad_gwish() function


double dpsi_ij(u_int i, u_int j, arma::mat& psi_mat, Rcpp::List& params) {

	arma::mat G     = params["G"];    // graph G represented as adjacency matrix
	u_int p         = params["p"];    // dimension of the graph G
	u_int b         = params["b"];    // degrees of freedom
	arma::vec nu_i  = params["nu_i"]; //
	// arma::mat L     = params["P"];

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
					d_ij += psi_mat(r,s) * dpsi(r, s, i, j, psi_mat, params);
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
					d_ij += psi_mat(r,s) * dpsi(r, s, i, j, psi_mat, params);
					// Rcpp::Rcout << "G[" << r+1 << ", " << s+1 << \
					//      "] = " << G(r,s) << std::endl;
				}
			} // end loop over s
		} // end loop over r
	} // end if-else
	// only get to this return statement if we go through the else()
	return d_ij + psi_mat(i,j);
} // end dpsi_ij() function


double dpsi(u_int r, u_int s, u_int i, u_int j,
	arma::mat& psi, Rcpp::List& params) {

	arma::mat G = params["G"];
	arma::mat L = params["P"];

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
			x += dpsi(r, k, i, j, psi, params) * xi(k, s, L); // correct G
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
			s0 += xi(k, s, L) * dpsi(r, k, i, j, psi, params);
		}
		for (k = 0; k < r; k++) {
			for (l = k; l < s; l++) {
				s10 += psi(k,l) * xi(l, s, L);
			}
			for (l = k; l < r; l++) {
				s11 += psi(k,l) * xi(l, r, L);
			}
			s1(k) = psi(k,r) * psi(k,s) +
				psi(k,r) * s10 + psi(k,s) * s11 + s10 * s11;
		} // end inner for()

		out = -s0 + 1 / std::pow(psi(r,r), 2) * arma::sum(s1);

	} else { // dpsi_rs / dpsi_ij
		// general case when derivative is wrt general (i, j), i > 0
		for (k = r; k < s; k++) {
			s0 += xi(k, s, L) * dpsi(r, k, i, j, psi, params);
		}
		/* calculation of s1, inner summation from 1:(r-1)
		   s1 = s10 + s11 + s12 + s13, where each component is one of the four
		   terms in the summation. s13 is consisted of 4 separate summations,
		   for which we use 2 for loops to compute.
		*/
		for (k = 0; k < r; k++) {

			// compute the intermediate summations:
			for (l = k; l < s; l++) {
				s110 += psi(k, l) * xi(l, s, L);
				// note: s111 is just s110 w/ derivative wrt (ij) applied to
				// the first term
				s111 += dpsi(k, l, i, j, psi, params) * xi(l, s, L);
			}
			for (l = k; l < r; l++) {
				s120 += psi(k, l) * xi(l, r, L);
				// note: s121 is just s120 w/ derivative wrt (ij) applied to
				// the first term
				s121 += dpsi(k, l, i, j, psi, params) * xi(l, r, L);
			}

			s10 = psi(k,s) * dpsi(k, r, i, j, psi, params) +
				psi(k,r) * dpsi(k, s, i, j, psi, params);
			s11 = dpsi(k, r, i, j, psi, params) * s110 + psi(k,r) * s111;
			s12 = dpsi(k, s, i, j, psi, params) * s120 + psi(k,s) * s121;
			s13 = s121 * s110 + s120 * s111;

			s1(k) = s10 + s11 + s12 + s13;
		} // end of for loop computing EACH TERM of s1 summation from 0:(r-1)
		out = -s0 - 1 / psi(r,r) * arma::sum(s1);
	} // end if-else()
	return out;
} // end dpsi() function


/** -------------------- end gradient functions ---------------------------- **/


/** -------------------- start hessian functions --------------------------- **/

// [[Rcpp::export]]
arma::mat hess_gwish(arma::mat& psi_mat, Rcpp::List& params) {

	u_int D           = params["D"];          // dimension of parameter space
	arma::mat G       = params["G"];          // graph G
	u_int n_nonfree   = params["n_nonfree"];  // number of nonfree elements
	arma::mat ind_mat = params["t_ind"];      // index of the free elements
	arma::mat vbar    = params["vbar"];       // index of nonfree elements
	u_int b           = params["b"];          // degrees of freedom
	arma::vec nu_i    = params["nu_i"];       // see Atay paper for definition
	// arma::mat psi_mat = create_psi_mat_cpp(u, params);

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
            x += std::pow(dpsi(rr, ss, i, j, psi_mat, params), 2) +
            psi_mat(rr,ss) * d2psi(rr, ss, i, j, i, j, psi_mat, params);

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
	            x += std::pow(dpsi(rr, ss, i, j, psi_mat, params), 2) +
	            	psi_mat(rr, ss) * d2psi(rr, ss, i, j, i, j, psi_mat, params);
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

	  		H(r,c) = dpsi(i, j, k, l, psi_mat, params) +
				d2psi_ijkl(i, j, k, l, psi_mat, params);
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
double d2psi_ijkl(u_int i, u_int j, u_int k, u_int l,
	arma::mat& psi, Rcpp::List& params) {

	u_int n_nonfree   = params["n_nonfree"];  // # nonfree elements
	arma::mat G       = params["G"];          // graph G
	arma::mat vbar    = params["vbar"];       // index of nonfree elements

	arma::vec sum_vbar(n_nonfree, arma::fill::zeros); // store summation terms
	u_int n, r, s;
	for (n = 0; n < n_nonfree; n++) {
		r = vbar(n, 0) - 1; // row index of nonfree element
		s = vbar(n, 1) - 1; // col index of nonfree element

		sum_vbar(n) = dpsi(r, s, k, l, psi, params) *
			dpsi(r, s, i, j, psi, params) +
			psi(r, s) * d2psi(r, s, i, j, k, l, psi, params);
	} // end for() over the nonfree elements
	return arma::sum(sum_vbar);
} // end d2psi_ijkl() function

double d2psi(u_int r, u_int s, u_int i, u_int j, u_int k, u_int l,
	arma::mat& psi, Rcpp::List& params) {

	u_int n_nonfree   = params["n_nonfree"];  // # nonfree elements
	arma::mat G       = params["G"];          // graph G
	arma::mat vbar    = params["vbar"];       // index of nonfree elements
	arma::mat L       = params["P"];          // UPPER cholesky factor

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
		return d2psi(r, s, k, l, i, j, psi, params);
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
		x0 += xi(m, s, L) * d2psi(r, m, i, j, k, l, psi, params);
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
				x2_0 += psi(m, n) * xi(n, s, L);
			}
			for (n = m; n < r; n++) {
				x3_0 += psi(m, n) * xi(n, r, L);
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
			x1 = psi(m, s) * dpsi(m, r, k, l, psi, params) +
				 psi(m, r) * dpsi(m, s, k, l, psi, params);
			double x2_0 = 0, x2_1 = 0;
			for (n = m; n < s; n++) {
				xi_ns = xi(n, s, L);
				x2_0 += xi_ns * psi(m, n);
				x2_1 += xi_ns * dpsi(m, n, k, l, psi, params);
			}
			x2 = dpsi(m, r, k, l, psi, params) * x2_0 + psi(m, r) * x2_1;

			double x3_0 = 0, x3_1 = 0;
			for (n = m; n < r; n++) {
				xi_nr = xi(n, r, L);
				x3_0 += xi_nr * psi(m, n);
				x3_1 += xi_nr * dpsi(m, n, k, l, psi, params);
			}
			x3 = dpsi(m, s, k, l, psi, params) * x3_0 + psi(m, s) * x3_1;

			// double x4_0 = 0, x4_1 = 0, x4_2 = 0, x4_3 = 0;
			x4 = x3_1 * x2_0 + x3_0 * x2_1;
			d2_sum_r(m) = 1/std::pow(psi(r,r), 2) * (x1 + x2 + x3 + x4);
		} else if ((r == k) && (k == l)) {
			// case 2: d^2(psi_rs) / (dpsi_ij dpsi_rr)
			// this case should no longer hit because we check this in the very
			// beginning, BEFORE any recursion begins
			Rcpp::Rcout << "case 2 -- swapping roles of (i,j) and (k,l)." << std::endl;
			Rcpp::Rcout << "(r,s) = (" << r << "," << s << "), " <<
				"(i,j) = (" << i << "," << j << "), " <<
				"(k,l) = (" << k << "," << l << ")" << std::endl;
			// just swap the roles of (i,j) and (k,l) and call d2psi() again
			// to hit the case above
			d2_sum_r(m) = d2psi(r, s, k, l, i, j, psi, params);
		} else { // case 3: dpsi(r,s) / dpsi(i,j) dpsi(k,l)
			// double x1 = 0, x2 = 0, x3 = 0, x4 = 0;
			x1 = psi(m, s) * d2psi(m, r, i, j, k, l, psi, params) +
				 dpsi(m, r, i, j, psi, params) * dpsi(m, s, k, l, psi, params) +
				 psi(m, r) * d2psi(m, s, i, j, k, l, psi, params) +
				 dpsi(m, r, k, l, psi, params) * dpsi(m, s, i, j, psi, params);

			/* compute x2 */
			double x2_0 = 0, x2_1 = 0, x2_2 = 0, x2_3 = 0;
			for (n = m; n < s; n++) { // check: ssummation over s
				xi_ns = xi(n, s, L);
				x2_0 += xi_ns * psi(m, n);
				x2_1 += xi_ns * dpsi(m, n, k, l, psi, params);
				x2_2 += xi_ns * dpsi(m, n, i, j, psi, params);
				x2_3 += xi_ns * d2psi(m, n, i, j, k, l, psi, params);
			}
			if (x2_1 == 0) {
				x2 = d2psi(m, r, i, j, k, l, psi, params) * x2_0 +
			     dpsi(m, r, k, l, psi, params) * x2_2 +
			     psi(m, r) * x2_3;
			} else if (x2_2 == 0) {
				x2 = d2psi(m, r, i, j, k, l, psi, params) * x2_0 +
			     dpsi(m, r, i, j, psi, params) * x2_1 +
			     psi(m, r) * x2_3;
			} else {
				x2 = d2psi(m, r, i, j, k, l, psi, params) * x2_0 +
			     dpsi(m, r, i, j, psi, params) * x2_1 +
			     dpsi(m, r, k, l, psi, params) * x2_2 +
			     psi(m, r) * x2_3;
			}
			/* compute x3 */
			double x3_0 = 0, x3_1 = 0, x3_2 = 0, x3_3 = 0;
			for (n = m; n < r; n++) { // check: summation over r
				xi_nr = xi(n, r, L);
				x3_0 += xi_nr * psi(m, n);
				x3_1 += xi_nr * dpsi(m, n, k, l, psi, params);
				x3_2 += xi_nr * dpsi(m, n, i, j, psi, params);
				x3_3 += xi_nr * d2psi(m, n, i, j, k, l, psi, params);
			}
			if (x3_1 == 0) {
				x3 = d2psi(m, s, i, j, k, l, psi, params)  * x3_0 +
			     dpsi(m, s, k, l, psi, params) * x3_2 +
			     psi(m, s) * x3_3;
			} else if (x3_2 == 0) {
				x3 = d2psi(m, s, i, j, k, l, psi, params)  * x3_0 +
			     dpsi(m, s, i, j, psi, params) * x3_1 +
			     psi(m, s) * x3_3;
			} else {
				x3 = d2psi(m, s, i, j, k, l, psi, params)  * x3_0 +
			     dpsi(m, s, i, j, psi, params) * x3_1 +
			     dpsi(m, s, k, l, psi, params) * x3_2 +
			     psi(m, s) * x3_3;
			}
			/* compute x4 */
			double x4_00 = 0, x4_01 = 0, x4_10 = 0, x4_11 = 0, x4_20 = 0,
				x4_21 = 0, x4_30 = 0, x4_31 = 0;
			x4_00 = x3_3; x4_10 = x3_2; x4_20 = x3_1; x4_30 = x3_0;
			x4_01 = x2_0; x4_11 = x2_1; x4_21 = x2_2; x4_31 = x2_3;
			/*
		    for (n = m; n < r; n++) {
				xi_nr = xi(n, r, L);
				x4_00 += xi_nr * d2psi(m, n, i, j, k, l, psi, params); // = x3_3
				x4_10 += xi_nr * dpsi(m, n, i, j, psi, params);        // = x3_2
				x4_20 += xi_nr * dpsi(m, n, k, l, psi, params);        // = x3_1
				x4_30 += xi_nr * psi(m, n);                            // = x3_0
			}
			for (n = m; n < s; n++) {
				xi_ns = xi(n, s, L);
				x4_01 += xi_ns * psi(m, n);                            // = x2_0
				x4_11 += xi_ns * dpsi(m, n, k, l, psi, params);        // = x2_1
				x4_21 += xi_ns * dpsi(m, n, i, j, psi, params);        // = x2_2
				x4_31 += xi_ns * d2psi(m, n, i, j, k, l, psi, params); // = x2_3
			}
			*/
			x4 = x4_00 * x4_01 + x4_10 * x4_11 + x4_20 * x4_21 + x4_30 * x4_31;
			d2_sum_r(m) = -1/psi(r,r) * (
				x1 + x2 + x3 + x4
			);
		}
	} // end for() over m, computing each term of the summation

	return - x0 + arma::sum(d2_sum_r);
} // end d2psi() function


/** -------------------- end hessian functions ----------------------------- **/