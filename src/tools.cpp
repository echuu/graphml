#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <cmath>
#include "graphml_types.h"

#define RCPP_ARMADILLO_RETURN_COLVEC_AS_VECTOR
// [[Rcpp::depends(RcppArmadillo)]]
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS



double xi(u_int i, u_int j, arma::mat& L) {
	// L is UPPER TRIANGULAR cholesky factor of the INVERSE scale matrix, i.e,
	// D^(-1) = L'L
	return L(i, j) / L(j, j);
} // end xi() function


// [[Rcpp::export]]
Rcpp::StringVector createDfName(unsigned int D) {
    Rcpp::Environment env = Rcpp::Environment::global_env();
    Rcpp::Function createDfName_R = env["createDfName_R"];
    Rcpp::StringVector nameVec = createDfName_R(D);
    return nameVec;
}


// [[Rcpp::export]]
Rcpp::DataFrame mat2df(arma::mat x, Rcpp::StringVector nameVec) {

    // convert x (J x (D+1)) a matrix to
    // x_df: with colnames: u1, u2, ... , uD, psi_u
    Rcpp::Environment env = Rcpp::Environment::global_env();
    Rcpp::Function mat2df_R = env["mat2df_R"];
    // Rcpp::StringVector nameVec = params["u_df_names"];
    Rcpp::DataFrame x_df = mat2df_R(x, nameVec);

    return x_df;
} // end of mat2df() function


/*
    fitTree(): returns an tree object, makes call to rpart::rpart() --
    this eventually will need to be ported to C++, waiting for Donald to finish
    that implementation
*/
// [[Rcpp::export]]
Rcpp::List fitTree(Rcpp::DataFrame x, Rcpp::Formula formula) {

    // Obtain environment containing function
    Rcpp::Environment base("package:rpart");
    Rcpp::Environment stats("package:stats");
    // Make function callable from C++
    Rcpp::Function cart = base["rpart"];
    Rcpp::Function model_frame = stats["model.frame"];
    // Call the function and receive its list (confirm?) output
    Rcpp::DataFrame x_mat = model_frame(Rcpp::_["formula"] = formula,
                                        Rcpp::_["data"] = x);

    Rcpp::List tree = cart(x_mat);

    return tree;
} // end fitTree() function

/* TODO: I dont' think we use this function anymore because I stuck it
directly into the approx_v1() function -- */
// [[Rcpp::export]]
Rcpp::List getPartition(Rcpp::List tree, arma::mat supp) {

    Rcpp::Environment tmp = Rcpp::Environment::global_env();
    Rcpp::Function f = tmp["extractPartitionSimple"];

    Rcpp::List partList = f(tree, supp);
    arma::mat part = partList["partition"];
    // Rcpp::Rcout<< part << std::endl;
    arma::vec leafId = partList["leaf_id"];
    int k = leafId.n_elem; // # of leaf nodes

    arma::vec locs = partList["locs"];
    // Rcpp::Rcout<< locs << std::endl;

    std::unordered_map<int, arma::vec> partitionMap;
    for (int i = 0; i < k; i++) {
        // add the leaf node's corresponding rectangle into the map
        int leaf_i = leafId(i);
        // arma::vec col_d = part[d];
        partitionMap[leaf_i] = part.col(i);
    }

    // Rcpp::Function unname = tmp["unname"];
    // Rcpp::Rcout<< unname(tree["where"]) << std::endl;

    return Rcpp::List::create(Rcpp::Named("locs") = locs,
						      Rcpp::Named("leafId") = leafId
            );
    // store first column as leaf_id
    // store the remaining columns as the lb/ub of the partition sets
    // create the map that defines the boundary here

    // return part;
} // end getPartition() function


// [[Rcpp::export]]
double lse(arma::vec arr, int count) {
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

   }
   else
   {
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


arma::mat getFreeElem(arma::umat G, u_int p) {
	arma::mat F = arma::conv_to<arma::mat>::from(G);
	for (u_int r = 1; r < p; r++) {
		for (u_int c = 0; c < r; c++) {
			F(r, c) = 0;
		}
	}
	return F;
} // end getFreeElem() function


arma::mat getNonFreeElem(arma::umat G, u_int p, u_int n_nonfree) {

	arma::mat F = arma::conv_to<arma::mat>::from(G);
	for (u_int r = 0; r < (p - 1); r++) {
		for (u_int c = r + 1; c < p; c++) {
			if (F(r, c) == 0) {
				F(r, c) = -1;
			}
		}
	}
	arma::uvec ind_vbar = find(F < 0); // extract indices of the nonfree elements
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


