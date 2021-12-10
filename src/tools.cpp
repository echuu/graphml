#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <cmath>
#include "graphml_types.h"

#define RCPP_ARMADILLO_RETURN_COLVEC_AS_VECTOR
// [[Rcpp::depends(RcppArmadillo)]]
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS



double xi(u_int i, u_int j, arma::mat& L) {
    // used to compute a fraction quantity in the gradient/hessian functions
	// L is UPPER TRIANGULAR cholesky factor of the INVERSE scale matrix, i.e,
	// D^(-1) = L'L
	return L(i, j) / L(j, j);
} // end xi() function


// [[Rcpp::export]]
Rcpp::StringVector createDfName(unsigned int D) {
    // this is called from initGraph() so that the column names of the samples
    // have names; these names are passed into mat2df() [see below] to create
    // the equivalent of u_df in the R implementation
    Rcpp::Environment env = Rcpp::Environment::global_env();
    Rcpp::Function createDfName_R = env["createDfName_R"];
    Rcpp::StringVector nameVec = createDfName_R(D);
    return nameVec;
}


// [[Rcpp::export]]
Rcpp::DataFrame mat2df(arma::mat x, Rcpp::StringVector nameVec) {
    // this function calls an R function to create the u_df dataframe that 
    // we need to pass into rpart() function; eventually when we have a C++ 
    // implementation for rpart, we will no longer need to create dataframes 
    // convert x (J x (D+1)) a matrix to
    // x_df: with colnames: u1, u2, ... , uD, psi_u
    Rcpp::Environment env = Rcpp::Environment::global_env();
    Rcpp::Function mat2df_R = env["mat2df_R"];
    // Rcpp::StringVector nameVec = params["u_df_names"];
    Rcpp::DataFrame x_df = mat2df_R(x, nameVec);

    return x_df;
} // end of mat2df() function


// [[Rcpp::export]]
double lse(arma::vec arr, int count) {
    // log-sum-exp() function
    if(count > 0){
        double maxVal = arr(0);
        double sum = 0;
        for (int i = 1 ; i < count ; i++){
            if (arr(i) > maxVal){
                maxVal = arr(i);
            }
        }
        for (int i = 0; i < count ; i++){
            sum += exp(arr(i) - maxVal);
        }
        return log(sum) + maxVal;
    } else {
        return 0.0;
    }
} // end of lse() function


arma::vec matrix2vector(arma::mat m, const bool byrow=false) {
  if (byrow) {
    return m.as_row();
  } else {
    return m.as_col();
  }
} // end matrix2vector() function



/** -------------------- gwish-specific functions -------------------------- **/

arma::mat getFreeElem(arma::umat G, u_int p) {
    // set lower diagonalt elements of G to 0 so that we have the free elements
    // on the diagonal + upper diagonal remaining; these are used to compute
    // intermediate quantities such as k_i, nu_i, b_i; see initGraph() for 
    // these calculations (see Atay paper for the 'A' matrix)
	arma::mat F = arma::conv_to<arma::mat>::from(G);
	for (u_int r = 1; r < p; r++) {
		for (u_int c = 0; c < r; c++) {
			F(r, c) = 0;
		}
	}
	return F;
} // end getFreeElem() function


arma::mat getNonFreeElem(arma::umat G, u_int p, u_int n_nonfree) {
    // get the 2-column matrix that has row, column index of each of the nonfree
    // elements
	arma::mat F = arma::conv_to<arma::mat>::from(G);
	for (u_int r = 0; r < (p - 1); r++) {
		for (u_int c = r + 1; c < p; c++) {
			if (F(r, c) == 0) {
				F(r, c) = -1;
			}
		}
	}
	arma::uvec ind_vbar = find(F < 0); // extract indices of the nonfree elmts
    // TODO: think about when n_nonfree == 0 --> what does this mean? what
    // happens to the subsequent calculations;
	arma::mat vbar(n_nonfree, 2, arma::fill::zeros);
	for (u_int n = 0; n < n_nonfree; n++) {
		vbar(n, 0) = ind_vbar(n) % p; // row of nonfree elmt
		vbar(n, 1) = ind_vbar(n) / p; // col of nonfree elmt
	}
	vbar = vbar + 1;

	return vbar;
} // end getFreeElem() function


/** ---------------------- reshaping functions ----------------------------- **/


/* vec2mat(): convert psi in vector form to matrix form, to include the non-
   free elements (represented as 0s)
*/ 
arma::mat vec2mat(arma::vec u, Rcpp::List& params) {

	u_int p           = params["p"];    // dimension of the graph G
	u_int D           = params["D"];    // dimension of parameter space
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
} // end vec2mat() function

