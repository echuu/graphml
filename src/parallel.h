#include <RcppArmadillo.h>
#include <algorithm>
#include <thread>
#include <functional>
#include <vector>

#define RCPP_ARMADILLO_RETURN_COLVEC_AS_VECTOR
// [[Rcpp::depends(RcppArmadillo)]]
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

arma::mat vec2mat(arma::vec u, u_int p, u_int D, u_int b,
                  arma::vec nu_i, arma::vec b_i, arma::mat P, arma::mat G,
                  arma::uvec ids);
double xi(u_int i, u_int j, arma::mat& L);

/* --------------------  gradient parallel functions  ----------------------- */
arma::vec grad_gwish_parallel(arma::mat psi_mat, arma::mat G, arma::uvec free,
                              u_int p, u_int D,
                              u_int b, arma::vec nu_i,
                              arma::mat L);

double dpsi_ij_parallel(u_int i, u_int j, arma::mat psi_mat,
                        arma::mat G, u_int p, u_int b, arma::vec nu_i, arma::mat L);

double dpsi_parallel(u_int r, u_int s, u_int i, u_int j, arma::mat psi,
                     arma::mat G, arma::mat L);

/* --------------------  hessian  parallel functions  ----------------------- */

arma::mat hess_gwish_parallel(arma::mat psi_mat, arma::mat G,
                              u_int D, u_int b, arma::vec nu_i,
                              arma::mat L, arma::mat ind_mat,
                              arma::mat vbar, u_int n_nonfree);

double d2psi_ijkl_parallel(u_int i, u_int j, u_int k, u_int l,
                           arma::mat psi, arma::mat G, arma::mat L, arma::mat vbar,
                           u_int n_nonfree);

double d2psi_parallel(u_int r, u_int s, u_int i, u_int j, u_int k, u_int l,
	                  arma::mat psi, arma::mat G, arma::mat L, arma::mat vbar,
                      u_int n_nonfree);




/// @param[in] nb_elements : size of your for loop
/// @param[in] functor(start, end) :
/// your function processing a sub chunk of the for loop.
/// "start" is the first index to process (included) until the index "end"
/// (excluded)
/// @code
///     for(int i = start; i < end; ++i)
///         computation(i);
/// @endcode
/// @param use_threads : enable / disable threads.
///
///
static
void parallel_for(unsigned nb_elements,
                  std::function<void (int start, int end)> functor,
                  bool use_threads = true)
{
    // -------
    unsigned nb_threads_hint = std::thread::hardware_concurrency();
    unsigned nb_threads = nb_threads_hint == 0 ? 8 : (nb_threads_hint);

    unsigned batch_size = nb_elements / nb_threads;
    unsigned batch_remainder = nb_elements % nb_threads;

    std::vector< std::thread > my_threads(nb_threads);

    if( use_threads )
    {
        // Multithread execution
        for(unsigned i = 0; i < nb_threads; ++i)
        {
            int start = i * batch_size;
            my_threads[i] = std::thread(functor, start, start+batch_size);
        }
    }
    else
    {
        // Single thread execution (for easy debugging)
        for(unsigned i = 0; i < nb_threads; ++i){
            int start = i * batch_size;
            functor( start, start+batch_size );
        }
    }

    // Deform the elements left
    int start = nb_threads * batch_size;
    functor( start, start+batch_remainder);

    // Wait for the other thread to finish their task
    if( use_threads )
        std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
}
