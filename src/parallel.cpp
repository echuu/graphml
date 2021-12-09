
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include "graphml_types.h"
#include "parallel.h"
#include <cmath>
#define RCPP_ARMADILLO_RETURN_COLVEC_AS_VECTOR
// [[Rcpp::depends(RcppArmadillo)]]
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS


/* functions that need to be parallelized:

    (1) create_psi_mat_cpp: shorten this name ffs
    (2) hess_gwish: extract params and pass directly into function
    (3) grad_gwish: extract params and pass directly into function

*/


double xi(u_int i, u_int j, arma::mat& L) {
	// L is UPPER TRIANGULAR cholesky factor of the INVERSE scale matrix, i.e,
	// D^(-1) = L'L
	return L(i, j) / L(j, j);
}

arma::mat hess_gwish_parallel(arma::mat psi_mat, arma::mat G,
                              u_int D, u_int b, arma::vec nu_i,
                              arma::mat L, arma::mat ind_mat,
                              arma::mat vbar, u_int n_nonfree) {

	// u_int D           = params["D"];          // dimension of parameter space
	// arma::mat G       = params["G"];          // graph G
	// u_int n_nonfree   = params["n_nonfree"];  // number of nonfree elements
	// arma::mat ind_mat = params["t_ind"];      // index of the free elements
	// arma::mat vbar    = params["vbar"];       // index of nonfree elements
	// u_int b           = params["b"];          // degrees of freedom
	// arma::vec nu_i    = params["nu_i"];       // see Atay paper for definition

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
            x += std::pow(dpsi_parallel(rr, ss, i, j, psi_mat, G, L), 2) +
            psi_mat(rr,ss) * d2psi_parallel(rr, ss, i, j, i, j, psi_mat, G, L, vbar, n_nonfree);

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
			   additional first order term, which reduces to 1 since dpsi_ij / dpsi_ij = 1
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
	            x += std::pow(dpsi_parallel(rr, ss, i, j, psi_mat, G, L), 2) +
	            	psi_mat(rr, ss) * d2psi_parallel(rr, ss, i, j, i, j, psi_mat, G, L, vbar, n_nonfree);
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

	  		H(r,c) = dpsi_parallel(i, j, k, l, psi_mat, G, L) +
				d2psi_ijkl_parallel(i, j, k, l, psi_mat, G, L, vbar, n_nonfree);
	    	H(c, r) = H(r, c); // reflect the value over the diagonal
	    } // end inner for() over upper triangular columns
	} // end for() population off-diagonal elements of the hessian


	return H;
} // end hess_gwish() function


double d2psi_ijkl_parallel(u_int i, u_int j, u_int k, u_int l,
	arma::mat psi, arma::mat G, arma::mat L, arma::mat vbar,
    u_int n_nonfree) {

	// u_int n_nonfree   = params["n_nonfree"];  // # nonfree elements
	// arma::mat G       = params["G"];          // graph G
	// arma::mat vbar    = params["vbar"];       // index of nonfree elements

	arma::vec sum_vbar(n_nonfree, arma::fill::zeros); // store summation terms
	u_int n, r, s;
	for (n = 0; n < n_nonfree; n++) {
		r = vbar(n, 0) - 1; // row index of nonfree element
		s = vbar(n, 1) - 1; // col index of nonfree element

		sum_vbar(n) = dpsi_parallel(r, s, k, l, psi, G, L) *
			dpsi_parallel(r, s, i, j, psi, G, L) +
			psi(r, s) * d2psi_parallel(r, s, i, j, k, l, psi, G, L, vbar, n_nonfree);
	} // end for() over the nonfree elements
	return arma::sum(sum_vbar);
} // end d2psi_ijkl() function


double d2psi_parallel(u_int r, u_int s, u_int i, u_int j, u_int k, u_int l,
	                  arma::mat psi, arma::mat G, arma::mat L, arma::mat vbar,
                      u_int n_nonfree) {

	// u_int n_nonfree   = params["n_nonfree"];  // # nonfree elements
	// arma::mat G       = params["G"];          // graph G
	// arma::mat vbar    = params["vbar"];       // index of nonfree elements
	// arma::mat L       = params["P"];          // UPPER cholesky factor

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
		return d2psi_parallel(r, s, k, l, i, j, psi, G, L, vbar, n_nonfree);
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
		x0 += xi(m, s, L) * d2psi_parallel(r, m, i, j, k, l, psi, G, L, vbar, n_nonfree);
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
			x1 = psi(m, s) * dpsi_parallel(m, r, k, l, psi, G, L) +
				 psi(m, r) * dpsi_parallel(m, s, k, l, psi, G, L);
			double x2_0 = 0, x2_1 = 0;
			for (n = m; n < s; n++) {
				xi_ns = xi(n, s, L);
				x2_0 += xi_ns * psi(m, n);
				x2_1 += xi_ns * dpsi_parallel(m, n, k, l, psi, G, L);
			}
			x2 = dpsi_parallel(m, r, k, l, psi, G, L) * x2_0 + psi(m, r) * x2_1;

			double x3_0 = 0, x3_1 = 0;
			for (n = m; n < r; n++) {
				xi_nr = xi(n, r, L);
				x3_0 += xi_nr * psi(m, n);
				x3_1 += xi_nr * dpsi_parallel(m, n, k, l, psi, G, L);
			}
			x3 = dpsi_parallel(m, s, k, l, psi, G, L) * x3_0 + psi(m, s) * x3_1;

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
			d2_sum_r(m) = d2psi_parallel(r, s, k, l, i, j, psi, G, L, vbar, n_nonfree);
		} else { // case 3: dpsi(r,s) / dpsi(i,j) dpsi(k,l)
			// double x1 = 0, x2 = 0, x3 = 0, x4 = 0;
			x1 = psi(m, s) * d2psi_parallel(m, r, i, j, k, l, psi, G, L, vbar, n_nonfree) +
				 dpsi_parallel(m, r, i, j, psi, G, L) * dpsi_parallel(m, s, k, l, psi, G, L) +
				 psi(m, r) * d2psi_parallel(m, s, i, j, k, l, psi, G, L, vbar, n_nonfree) +
				 dpsi_parallel(m, r, k, l, psi, G, L) * dpsi_parallel(m, s, i, j, psi, G, L);

			/* compute x2 */
			double x2_0 = 0, x2_1 = 0, x2_2 = 0, x2_3 = 0;
			for (n = m; n < s; n++) { // check: ssummation over s
				xi_ns = xi(n, s, L);
				x2_0 += xi_ns * psi(m, n);
				x2_1 += xi_ns * dpsi_parallel(m, n, k, l, psi, G, L);
				x2_2 += xi_ns * dpsi_parallel(m, n, i, j, psi, G, L);
				x2_3 += xi_ns * d2psi_parallel(m, n, i, j, k, l, psi, G, L, vbar, n_nonfree);
			}
			if (x2_1 == 0) {
				x2 = d2psi_parallel(m, r, i, j, k, l, psi, G, L, vbar, n_nonfree) * x2_0 +
			     dpsi_parallel(m, r, k, l, psi, G, L) * x2_2 +
			     psi(m, r) * x2_3;
			} else if (x2_2 == 0) {
				x2 = d2psi_parallel(m, r, i, j, k, l, psi, G, L, vbar, n_nonfree) * x2_0 +
			     dpsi_parallel(m, r, i, j, psi, G, L) * x2_1 +
			     psi(m, r) * x2_3;
			} else {
				x2 = d2psi_parallel(m, r, i, j, k, l, psi, G, L, vbar, n_nonfree) * x2_0 +
			     dpsi_parallel(m, r, i, j, psi, G, L) * x2_1 +
			     dpsi_parallel(m, r, k, l, psi, G, L) * x2_2 +
			     psi(m, r) * x2_3;
			}
			/* compute x3 */
			double x3_0 = 0, x3_1 = 0, x3_2 = 0, x3_3 = 0;
			for (n = m; n < r; n++) { // check: summation over r
				xi_nr = xi(n, r, L);
				x3_0 += xi_nr * psi(m, n);
				x3_1 += xi_nr * dpsi_parallel(m, n, k, l, psi, G, L);
				x3_2 += xi_nr * dpsi_parallel(m, n, i, j, psi, G, L);
				x3_3 += xi_nr * d2psi_parallel(m, n, i, j, k, l, psi, G, L, vbar, n_nonfree);
			}
			if (x3_1 == 0) {
				x3 = d2psi_parallel(m, s, i, j, k, l, psi, G, L, vbar, n_nonfree)  * x3_0 +
			     dpsi_parallel(m, s, k, l, psi, G, L) * x3_2 +
			     psi(m, s) * x3_3;
			} else if (x3_2 == 0) {
				x3 = d2psi_parallel(m, s, i, j, k, l, psi, G, L, vbar, n_nonfree)  * x3_0 +
			     dpsi_parallel(m, s, i, j, psi, G, L) * x3_1 +
			     psi(m, s) * x3_3;
			} else {
				x3 = d2psi_parallel(m, s, i, j, k, l, psi, G, L, vbar, n_nonfree)  * x3_0 +
			     dpsi_parallel(m, s, i, j, psi, G, L) * x3_1 +
			     dpsi_parallel(m, s, k, l, psi, G, L) * x3_2 +
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
} // end d2psi_parallel() function





/* --------------------  gradient parallel functions  ----------------------- */

// [[Rcpp::export]]
arma::vec grad_gwish_parallel(arma::mat psi_mat, arma::mat G, arma::uvec free,
                              u_int p, u_int D,
                              u_int b, arma::vec nu_i,
                              arma::mat L) {

    // commented out for parallelization
	// arma::mat G       = params["G"]; // graph G represented as adjacency matrix
	// u_int p           = params["p"]; // dimension of the graph G
	// arma::uvec free   = params["free_index"];
	// u_int D           = params["D"]; // dimension of parameter space


	// TODO: implement create_psi_mat() function later; for now, we pass it in
	// arma::mat psi_mat = vec2chol(u, p)
	// arma::mat psi_mat = create_psi_mat_cpp(u, params);

	// initialize matrix that can store the gradient elements
	arma::mat gg(p, p, arma::fill::zeros);
	// populate the gradient matrix entry by entry
	for (u_int i = 0; i < p; i++) {
		for (u_int j = i; j < p; j++) {
			if (G(i,j) > 0) {
				gg(i,j) = dpsi_ij_parallel(i, j, psi_mat, G, p, b, nu_i, L);
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


double dpsi_ij_parallel(u_int i, u_int j, arma::mat psi_mat,
                        arma::mat G, u_int p, u_int b, arma::vec nu_i,
                        arma::mat L) {

	// arma::mat G     = params["G"];    // graph G represented as adjacency matrix
	// u_int p         = params["p"];    // dimension of the graph G
	// u_int b         = params["b"];    // degrees of freedom
	// arma::vec nu_i  = params["nu_i"]; //

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
					d_ij += psi_mat(r,s) * dpsi_parallel(r, s, i, j, psi_mat, G, L);
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
					d_ij += psi_mat(r,s) * dpsi_parallel(r, s, i, j, psi_mat, G, L);
					// Rcpp::Rcout << "G[" << r+1 << ", " << s+1 << \
					//      "] = " << G(r,s) << std::endl;
				}
			} // end loop over s
		} // end loop over r
	} // end if-else
	// only get to this return statement if we go through the else()
	return d_ij + psi_mat(i,j);
} // end dpsi_ij() function


double dpsi_parallel(u_int r, u_int s, u_int i, u_int j, arma::mat psi,
                     arma::mat G, arma::mat L) {

	// arma::mat G = params["G"];
	// arma::mat L = params["P"];

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
			x += dpsi_parallel(r, k, i, j, psi, G, L) * xi(k, s, L); // correct G
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
			s0 += xi(k, s, L) * dpsi_parallel(r, k, i, j, psi, G, L);
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
			s0 += xi(k, s, L) * dpsi_parallel(r, k, i, j, psi, G, L);
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
				s111 += dpsi_parallel(k, l, i, j, psi, G, L) * xi(l, s, L);
			}
			for (l = k; l < r; l++) {
				s120 += psi(k, l) * xi(l, r, L);
				// note: s121 is just s120 w/ derivative wrt (ij) applied to
				// the first term
				s121 += dpsi_parallel(k, l, i, j, psi, G, L) * xi(l, r, L);
			}

			s10 = psi(k,s) * dpsi_parallel(k, r, i, j, psi, G, L) +
				psi(k,r) * dpsi_parallel(k, s, i, j, psi, G, L);
			s11 = dpsi_parallel(k, r, i, j, psi, G, L) * s110 + psi(k,r) * s111;
			s12 = dpsi_parallel(k, s, i, j, psi, G, L) * s120 + psi(k,s) * s121;
			s13 = s121 * s110 + s120 * s111;

			s1(k) = s10 + s11 + s12 + s13;
		} // end of for loop computing EACH TERM of s1 summation from 0:(r-1)
		out = -s0 - 1 / psi(r,r) * arma::sum(s1);
	} // end if-else()
	return out;
} // end dpsi_parallel() function




// [[Rcpp::export]]
arma::mat vec2mat(arma::vec u, u_int p, u_int D, u_int b,
                  arma::vec nu_i, arma::vec b_i, arma::mat P, arma::mat G,
                  arma::uvec ids) {

	// u_int p           = params["p"];    // dimension of the graph G
	// u_int D           = params["D"];    // dimension of parameter space
	// u_int b           = params["b"];    // degrees of freedom
	// arma::vec nu_i    = params["nu_i"]; // see p. 329 of Atay (step 2)
	// arma::vec b_i     = params["b_i"];  // see p. 329 of Atay (step 2)
	// arma::mat P       = params["P"];    // upper cholesky factor of V_n
	// arma::mat G       = params["G"];    // graph G

	// arma::vec edgeInd = params["edgeInd"];
	//Rcpp::Rcout << "hello" << std::endl;

	/* convert u into the matrix version with free-elements populated */
	// boolean vectorized version of G
	//arma::vec G_bool  = params["FREE_PARAMS_ALL"];
	// arma::uvec ids =  params["free_index"];
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
} // end vec2mat() function
