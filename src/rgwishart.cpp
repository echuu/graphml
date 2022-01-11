#include <R.h>
#include <Rmath.h>

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <Rcpp.h>
#include <cmath>

#include <vector>
using std::vector;

#include <iostream>
using std::cout;
using std::endl;

arma::mat rwish_c(arma::mat Ts, unsigned int b, unsigned int p);
arma::mat rgwish_c(arma::mat G, arma::mat Ts, unsigned int b, unsigned int p,
    double threshold_c);
arma::mat rgw(unsigned int J, Rcpp::List& obj);



arma::mat rgw(unsigned int J, Rcpp::List& obj) {

    arma::mat G     = obj["G"];
    arma::mat V     = obj["V"];
    arma::mat P_inv = obj["P_inv"];
    arma::mat P     = obj["P"];
	  unsigned int b  = obj["b"];
	  unsigned int p  = obj["p"];
    unsigned int D  = obj["D"];
    arma::mat samps(D, J, arma::fill::zeros);
    arma::mat omega, phi, zeta;
    arma::vec u0, u;
	  arma::uvec ids  = obj["free_index"];

    for (unsigned int j = 0; j < J; j++) {
        omega = rgwish_c(G, P, b, p, 1e-8);     // draw covar matrix
        phi   = arma::chol(omega);              // upper choleksy
        zeta  = phi * P_inv;                    // compute transformation
        u0    = arma::vectorise(zeta);
        u     = u0(ids);                        // extract free elements
        samps.col(j) = u;
    } // end sampling loop
	return samps.t();
} // end rgw() function


/* sampling from Wishart distribution, in which Ts = chol( solve( Ds ) ) */
arma::mat rwish_c(arma::mat Ts, unsigned int b, unsigned int p)
// Ts upper triangle, psi lower triangle
{
  arma::mat psi(p, p, arma::fill::zeros);

  for(unsigned int i=0; i<p; i++)
    psi(i, i) = sqrt( R::rgamma( ( b + p - i - 1 ) / 2.0, 2.0 ) );

  for(unsigned int j=1; j<p; j++)
    for(unsigned int i=0; i<j; i++)
      psi(j, i) = R::rnorm(0,1);

  arma::mat C = psi.t() * Ts;
  arma::mat K = C.t() * C;
  // arma::mat K = Ts.t() * psi * psi.t() * Ts;
  return K;
} // end rwish_c() function



/* sampling from G-Wishart distribution */
// G is adjacency matrix which has zero in its diagonal, threshold = 1e-8
arma::mat rgwish_c(arma::mat G, arma::mat Ts, unsigned int b, unsigned int p,
    double threshold_c)
// Ts upper triangle
{
  unsigned int l, size_node;
  // double threshold_c = 1E-8;
  arma::mat K = rwish_c(Ts, b, p);
  arma::mat sigma_start = arma::inv(K);
  arma::mat sigma = sigma_start;
  arma::mat sigma_last(p, p, arma::fill::zeros);
  arma::vec beta_star(p, arma::fill::zeros);
  arma::vec sigma_start_i(p, arma::fill::zeros);

  // arma::vec sigma_start_N_i(p, arma::fill::zeros);
  // arma::uvec N_i(p, arma::fill::zeros);
  // arma::mat sigma_N_i(p, p, arma::fill::zeros);

  double mean_diff = 1.0;
  while(mean_diff > threshold_c)
  {
    sigma_last = sigma;

    for(unsigned int i=0; i<p; i++)
    {
      size_node = 0;
      for(unsigned int j=0; j<p; j++) size_node += G(j, i);

      if(size_node>0)
      {
        arma::mat sigma_N_i(size_node, size_node, arma::fill::zeros);
        arma::vec sigma_start_N_i(size_node, arma::fill::zeros);
        arma::uvec N_i(size_node, arma::fill::zeros);

        l = 0;
        for(unsigned int j=0; j<p; j++)
        {
          if( G(j, i) )
          {
            sigma_start_N_i(l) = sigma_start(i, j);
            N_i(l++) = j;
          }else
            beta_star(j) = 0.0;
        }

        for(unsigned int ii=0; ii<size_node; ii++)
          for(unsigned int jj=0; jj<size_node; jj++)
            sigma_N_i(ii, jj) = sigma( N_i(ii), N_i(jj) );

        sigma_start_N_i = arma::inv(sigma_N_i) * sigma_start_N_i;
        for(unsigned int j=0; j<size_node; j++) beta_star( N_i(j) ) = sigma_start_N_i(j);
        sigma_start_i = sigma * beta_star;

        for(unsigned int j=0; j<i; j++)
        {
          sigma(j, i) = sigma_start_i(j);
          sigma(i, j) = sigma_start_i(j);
        }

        for(unsigned int j=i+1; j<p; j++)
        {
          sigma(j, i) = sigma_start_i(j);
          sigma(i, j) = sigma_start_i(j);
        }
      }
      else{
        for(unsigned int j=0; j<i; j++)
        {
          sigma(j, i) = 0.0;
          sigma(i, j) = 0.0;
        }

        for(unsigned int j=i+1; j<p; j++)
        {
          sigma(j, i) = 0.0;
          sigma(i, j) = 0.0;
        }
      }
    }

    mean_diff = arma::accu(arma::abs(sigma - sigma_last));
    mean_diff /= p*p;
  }

  K = inv(sigma);
  return K;
} // end rgwish_c() function
