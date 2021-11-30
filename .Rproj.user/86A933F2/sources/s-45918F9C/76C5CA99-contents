

#include <RcppArmadillo.h>
#include <cmath>

using namespace Rcpp;

typedef unsigned int u_int;

// [[Rcpp::depends(RcppArmadillo)]]
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

inline float psi(Rcpp::NumericVector& u, Rcpp::List& params);
inline float loglike(Rcpp::NumericVector& u, Rcpp::List& params);
inline float logprior(Rcpp::NumericVector& u, Rcpp::List& params);
inline arma::vec grad(Rcpp::NumericVector& u, Rcpp::List& params);
inline arma::mat hess(Rcpp::NumericVector& u, Rcpp::List& params);

// [[Rcpp::export]]
inline float psi(Rcpp::NumericVector& u, Rcpp::List& params) {
	return -loglike(u, params) - logprior(u, params);
}



// [[Rcpp::export]]
inline float loglike(Rcpp::NumericVector& u, Rcpp::List& params) {

	u_int n     = params["n"];
	arma::vec y = params["y"];
  arma::mat X = params["X"];

	u_int D = u.size() - 1;
	u_int d = D; // dimension of beta

	float tau = u[D];
    arma::vec beta = u[Range(0, D-1)];

    arma::vec z = y - X * beta;
    float ll = -0.5 * n * std::log(2.0 * M_PI) + 0.5 * n * std::log(tau) -
    	0.5 * tau * arma::as_scalar(z.t() * z);

	return ll;
}



// [[Rcpp::export]]
inline float logprior(Rcpp::NumericVector& u, Rcpp::List& params) {

	arma::vec beta0 = params["beta0"];  // pre-computed quantity
	float alpha     = params["alpha"];  // shape
	float delta     = params["delta"];  // rate
	arma::mat tau0  = params["tau0"];   // prior precision matrix
	float ldtau0    = params["ldtau0"]; // log(det(tau0))

	float lp; // store logprior result

	u_int D = u.size() - 1;
	u_int d = D; // dimension of beta
    float tau = u[D];
    arma::vec beta = u[Range(0, D-1)];

	arma::vec dist = beta - beta0;

	float lgamma_pdf = 0.5 * alpha * std::log(delta / 2) - lgamma(alpha / 2) -
		delta * tau / 2 + (0.5 * alpha - 1) * std::log(tau);

	lp = -0.5 * d * std::log(2.0 * M_PI) + 0.5 * d * std::log(tau) +
		0.5 * ldtau0 + lgamma_pdf -
		0.5 * tau * arma::as_scalar(dist.t() * tau0 * dist);

	return lp;
}



// [[Rcpp::export]]
inline arma::vec grad(Rcpp::NumericVector& u, Rcpp::List& params) {

	arma::vec beta0 = params["beta0"];  // pre-computed quantity
	arma::vec mu0   = params["mu0"];    // pre-computed quantity
	arma::mat tau0  = params["tau0"];   // prior precision matrix
	arma::mat X     = params["X"];
	arma::vec y     = params["y"];
	float alpha     = params["alpha"];  // shape
	float delta     = params["delta"];  // rate
	u_int n         = params["n"];

	u_int D = u.size() - 1;
	u_int d = D; // dimension of beta
    float tau = u[D];
    arma::vec beta = u[Range(0, D-1)];

	arma::vec diff = beta - mu0;
	arma::vec yxb  = (y - X * beta);

	arma::vec g(u.size());

	arma::uvec ind(d);
	ind = arma::regspace<arma::uvec>(0, g.n_elem-2);

	arma::uvec ind_tau(1);
	ind_tau(0) = d;


	// Rcpp::NumericVector g(u.size());
	g.elem(ind) = tau * (tau0 * diff - X.t() * yxb);

	g(d) = -1 / tau * (0.5 * (n + d + alpha) - 1) +
			0.5 * (delta + arma::as_scalar(yxb.t() * yxb) +
                   arma::as_scalar(diff.t() * tau0 * diff));
	return g;
}



// [[Rcpp::export]]
inline arma::mat hess(Rcpp::NumericVector& u, Rcpp::List& params) {

	arma::vec mu0     = params["mu0"];
	arma::vec Xty     = params["Xty"];
	arma::mat M       = params["M"];
	arma::mat Lambda0 = params["tau0"];
	float alpha       = params["alpha"];  // shape
	u_int n           = params["n"];

	u_int D = u.size();      // dimension of theta
	u_int d = D - 1;         // dimension of beta
	double tau = u[D - 1];
    arma::vec beta = u[Range(0, d - 1)];

	arma::mat H(D, D, arma::fill::zeros);
	arma::mat h11(d, d, arma::fill::zeros); // hessian for beta component
	arma::vec h12(d, arma::fill::zeros);
	double h22;

	h11 = tau * M;
	h12 = M * beta - Xty - Lambda0 * mu0;
	h22 = 1 / std::pow(tau, 2) * (0.5 * (n + d + alpha) - 1);
	H.submat(0, d, d-1, d) = h12;
	H.submat(d, 0, d, d-1) = h12.t();
	H.submat(0, 0, d - 1, d - 1) = h11;
	H(d, d) = h22;

	return H;

}



