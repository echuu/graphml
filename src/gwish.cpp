
#include "gwish.h"		 // this file includes density.h : psi_cpp, grad, hess
#include "rgwishart.h"
#include "partition.h"
#include "tools.h"
#include "epmgp.h"
#include <cmath>

#include "Node.h"
#include "Graph.h"
#include "Tree.h"
#include "Interval.h"
#define RCPP_ARMADILLO_RETURN_COLVEC_AS_VECTOR

// [[Rcpp::depends(RcppArmadillo)]]
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

using namespace Rcpp;


/* ------------------  algorithm specific functions ------------------------- */


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



/* ---------------  end of algorithm specific functions --------------------- */


// end gwish.cpp file
