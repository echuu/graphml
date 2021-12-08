// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "graphml_types.h"
#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// ep_logz
double ep_logz(arma::vec m, arma::mat K, arma::vec lb, arma::vec ub);
RcppExport SEXP _graphml_ep_logz(SEXP mSEXP, SEXP KSEXP, SEXP lbSEXP, SEXP ubSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type m(mSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lb(lbSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type ub(ubSEXP);
    rcpp_result_gen = Rcpp::wrap(ep_logz(m, K, lb, ub));
    return rcpp_result_gen;
END_RCPP
}
// getJT
Rcpp::List getJT(arma::umat EdgeMat);
RcppExport SEXP _graphml_getJT(SEXP EdgeMatSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::umat >::type EdgeMat(EdgeMatSEXP);
    rcpp_result_gen = Rcpp::wrap(getJT(EdgeMat));
    return rcpp_result_gen;
END_RCPP
}
// log_exp_mc
arma::vec log_exp_mc(arma::umat G, arma::uvec nu, unsigned int b, arma::mat H, unsigned int check_H, unsigned int mc, unsigned int p);
RcppExport SEXP _graphml_log_exp_mc(SEXP GSEXP, SEXP nuSEXP, SEXP bSEXP, SEXP HSEXP, SEXP check_HSEXP, SEXP mcSEXP, SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::umat >::type G(GSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type b(bSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type H(HSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type check_H(check_HSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type mc(mcSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(log_exp_mc(G, nu, b, H, check_H, mc, p));
    return rcpp_result_gen;
END_RCPP
}
// gnorm_c
double gnorm_c(arma::umat Adj, double b, arma::mat D, unsigned int iter);
RcppExport SEXP _graphml_gnorm_c(SEXP AdjSEXP, SEXP bSEXP, SEXP DSEXP, SEXP iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::umat >::type Adj(AdjSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type D(DSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type iter(iterSEXP);
    rcpp_result_gen = Rcpp::wrap(gnorm_c(Adj, b, D, iter));
    return rcpp_result_gen;
END_RCPP
}
// log_multi_gamma
double log_multi_gamma(int p, double n);
RcppExport SEXP _graphml_log_multi_gamma(SEXP pSEXP, SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< double >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(log_multi_gamma(p, n));
    return rcpp_result_gen;
END_RCPP
}
// log_wishart_norm
double log_wishart_norm(int p, double b, arma::mat D);
RcppExport SEXP _graphml_log_wishart_norm(SEXP pSEXP, SEXP bSEXP, SEXP DSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type D(DSEXP);
    rcpp_result_gen = Rcpp::wrap(log_wishart_norm(p, b, D));
    return rcpp_result_gen;
END_RCPP
}
// gnormJT
double gnormJT(arma::umat Adj, arma::umat EdgeMat, double b, arma::mat D, int iter);
RcppExport SEXP _graphml_gnormJT(SEXP AdjSEXP, SEXP EdgeMatSEXP, SEXP bSEXP, SEXP DSEXP, SEXP iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::umat >::type Adj(AdjSEXP);
    Rcpp::traits::input_parameter< arma::umat >::type EdgeMat(EdgeMatSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type D(DSEXP);
    Rcpp::traits::input_parameter< int >::type iter(iterSEXP);
    rcpp_result_gen = Rcpp::wrap(gnormJT(Adj, EdgeMat, b, D, iter));
    return rcpp_result_gen;
END_RCPP
}
// lse
double lse(arma::vec arr, int count);
RcppExport SEXP _graphml_lse(SEXP arrSEXP, SEXP countSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type arr(arrSEXP);
    Rcpp::traits::input_parameter< int >::type count(countSEXP);
    rcpp_result_gen = Rcpp::wrap(lse(arr, count));
    return rcpp_result_gen;
END_RCPP
}
// matrix2vector
arma::vec matrix2vector(arma::mat m, const bool byrow);
RcppExport SEXP _graphml_matrix2vector(SEXP mSEXP, SEXP byrowSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type m(mSEXP);
    Rcpp::traits::input_parameter< const bool >::type byrow(byrowSEXP);
    rcpp_result_gen = Rcpp::wrap(matrix2vector(m, byrow));
    return rcpp_result_gen;
END_RCPP
}
// getFreeElem
arma::mat getFreeElem(arma::umat G, u_int p);
RcppExport SEXP _graphml_getFreeElem(SEXP GSEXP, SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::umat >::type G(GSEXP);
    Rcpp::traits::input_parameter< u_int >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(getFreeElem(G, p));
    return rcpp_result_gen;
END_RCPP
}
// getNonFreeElem
arma::mat getNonFreeElem(arma::umat G, u_int p, u_int n_nonfree);
RcppExport SEXP _graphml_getNonFreeElem(SEXP GSEXP, SEXP pSEXP, SEXP n_nonfreeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::umat >::type G(GSEXP);
    Rcpp::traits::input_parameter< u_int >::type p(pSEXP);
    Rcpp::traits::input_parameter< u_int >::type n_nonfree(n_nonfreeSEXP);
    rcpp_result_gen = Rcpp::wrap(getNonFreeElem(G, p, n_nonfree));
    return rcpp_result_gen;
END_RCPP
}
// init_graph
Rcpp::List init_graph(arma::umat G, u_int b, arma::mat V);
RcppExport SEXP _graphml_init_graph(SEXP GSEXP, SEXP bSEXP, SEXP VSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::umat >::type G(GSEXP);
    Rcpp::traits::input_parameter< u_int >::type b(bSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type V(VSEXP);
    rcpp_result_gen = Rcpp::wrap(init_graph(G, b, V));
    return rcpp_result_gen;
END_RCPP
}
// evalPsi
arma::mat evalPsi(arma::mat samps, Rcpp::List& params);
RcppExport SEXP _graphml_evalPsi(SEXP sampsSEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type samps(sampsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(evalPsi(samps, params));
    return rcpp_result_gen;
END_RCPP
}
// approx_v1
double approx_v1(Rcpp::DataFrame u_df, arma::vec uStar, arma::mat data, Rcpp::List& params);
RcppExport SEXP _graphml_approx_v1(SEXP u_dfSEXP, SEXP uStarSEXP, SEXP dataSEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type u_df(u_dfSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type uStar(uStarSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type data(dataSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(approx_v1(u_df, uStar, data, params));
    return rcpp_result_gen;
END_RCPP
}
// generalApprox
double generalApprox(arma::umat G, u_int b, arma::mat V, u_int J);
RcppExport SEXP _graphml_generalApprox(SEXP GSEXP, SEXP bSEXP, SEXP VSEXP, SEXP JSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::umat >::type G(GSEXP);
    Rcpp::traits::input_parameter< u_int >::type b(bSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type V(VSEXP);
    Rcpp::traits::input_parameter< u_int >::type J(JSEXP);
    rcpp_result_gen = Rcpp::wrap(generalApprox(G, b, V, J));
    return rcpp_result_gen;
END_RCPP
}
// approxWrapper
double approxWrapper(arma::mat data, arma::vec locs, arma::vec uStar, u_int D, arma::mat bounds, arma::vec leafId, Rcpp::List& params);
RcppExport SEXP _graphml_approxWrapper(SEXP dataSEXP, SEXP locsSEXP, SEXP uStarSEXP, SEXP DSEXP, SEXP boundsSEXP, SEXP leafIdSEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type data(dataSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type locs(locsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type uStar(uStarSEXP);
    Rcpp::traits::input_parameter< u_int >::type D(DSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type bounds(boundsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type leafId(leafIdSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(approxWrapper(data, locs, uStar, D, bounds, leafId, params));
    return rcpp_result_gen;
END_RCPP
}
// calcMode
arma::vec calcMode(arma::mat u_df, Rcpp::List& params);
RcppExport SEXP _graphml_calcMode(SEXP u_dfSEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type u_df(u_dfSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(calcMode(u_df, params));
    return rcpp_result_gen;
END_RCPP
}
// approx_integral
double approx_integral(u_int K, arma::mat& psi_df, arma::mat& bounds, Rcpp::List& params);
RcppExport SEXP _graphml_approx_integral(SEXP KSEXP, SEXP psi_dfSEXP, SEXP boundsSEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< u_int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type psi_df(psi_dfSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type bounds(boundsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(approx_integral(K, psi_df, bounds, params));
    return rcpp_result_gen;
END_RCPP
}
// create_psi_mat_cpp
arma::mat create_psi_mat_cpp(arma::vec u, Rcpp::List& params);
RcppExport SEXP _graphml_create_psi_mat_cpp(SEXP uSEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type u(uSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(create_psi_mat_cpp(u, params));
    return rcpp_result_gen;
END_RCPP
}
// psi_cpp
double psi_cpp(arma::vec& u, Rcpp::List& params);
RcppExport SEXP _graphml_psi_cpp(SEXP uSEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type u(uSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(psi_cpp(u, params));
    return rcpp_result_gen;
END_RCPP
}
// psi_cpp_mat
double psi_cpp_mat(arma::mat& psi_mat, Rcpp::List& params);
RcppExport SEXP _graphml_psi_cpp_mat(SEXP psi_matSEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type psi_mat(psi_matSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(psi_cpp_mat(psi_mat, params));
    return rcpp_result_gen;
END_RCPP
}
// grad_gwish
arma::vec grad_gwish(arma::mat& psi_mat, Rcpp::List& params);
RcppExport SEXP _graphml_grad_gwish(SEXP psi_matSEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type psi_mat(psi_matSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(grad_gwish(psi_mat, params));
    return rcpp_result_gen;
END_RCPP
}
// hess_gwish
arma::mat hess_gwish(arma::mat& psi_mat, Rcpp::List& params);
RcppExport SEXP _graphml_hess_gwish(SEXP psi_matSEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type psi_mat(psi_matSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(hess_gwish(psi_mat, params));
    return rcpp_result_gen;
END_RCPP
}
// support
arma::mat support(arma::mat samps, u_int D);
RcppExport SEXP _graphml_support(SEXP sampsSEXP, SEXP DSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type samps(sampsSEXP);
    Rcpp::traits::input_parameter< u_int >::type D(DSEXP);
    rcpp_result_gen = Rcpp::wrap(support(samps, D));
    return rcpp_result_gen;
END_RCPP
}
// findAllCandidatePoints
std::unordered_map<int, arma::vec> findAllCandidatePoints(arma::mat data, arma::vec locs, arma::vec uStar, u_int D);
RcppExport SEXP _graphml_findAllCandidatePoints(SEXP dataSEXP, SEXP locsSEXP, SEXP uStarSEXP, SEXP DSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type data(dataSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type locs(locsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type uStar(uStarSEXP);
    Rcpp::traits::input_parameter< u_int >::type D(DSEXP);
    rcpp_result_gen = Rcpp::wrap(findAllCandidatePoints(data, locs, uStar, D));
    return rcpp_result_gen;
END_RCPP
}
// createPartitionMap
std::unordered_map<int, arma::vec> createPartitionMap(arma::mat bounds, arma::vec leafId);
RcppExport SEXP _graphml_createPartitionMap(SEXP boundsSEXP, SEXP leafIdSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type bounds(boundsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type leafId(leafIdSEXP);
    rcpp_result_gen = Rcpp::wrap(createPartitionMap(bounds, leafId));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_hello_world
arma::mat rcpparma_hello_world();
RcppExport SEXP _graphml_rcpparma_hello_world() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(rcpparma_hello_world());
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_outerproduct
arma::mat rcpparma_outerproduct(const arma::colvec& x);
RcppExport SEXP _graphml_rcpparma_outerproduct(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_outerproduct(x));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_innerproduct
double rcpparma_innerproduct(const arma::colvec& x);
RcppExport SEXP _graphml_rcpparma_innerproduct(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_innerproduct(x));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_bothproducts
Rcpp::List rcpparma_bothproducts(const arma::colvec& x);
RcppExport SEXP _graphml_rcpparma_bothproducts(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_bothproducts(x));
    return rcpp_result_gen;
END_RCPP
}
// rgw
arma::mat rgw(unsigned int J, Rcpp::List& obj);
RcppExport SEXP _graphml_rgw(SEXP JSEXP, SEXP objSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type J(JSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type obj(objSEXP);
    rcpp_result_gen = Rcpp::wrap(rgw(J, obj));
    return rcpp_result_gen;
END_RCPP
}
// rwish_c
arma::mat rwish_c(arma::mat Ts, unsigned int b, unsigned int p);
RcppExport SEXP _graphml_rwish_c(SEXP TsSEXP, SEXP bSEXP, SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type Ts(TsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type b(bSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(rwish_c(Ts, b, p));
    return rcpp_result_gen;
END_RCPP
}
// rgwish_c
arma::mat rgwish_c(arma::mat G, arma::mat Ts, unsigned int b, unsigned int p, double threshold_c);
RcppExport SEXP _graphml_rgwish_c(SEXP GSEXP, SEXP TsSEXP, SEXP bSEXP, SEXP pSEXP, SEXP threshold_cSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type G(GSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Ts(TsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type b(bSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type p(pSEXP);
    Rcpp::traits::input_parameter< double >::type threshold_c(threshold_cSEXP);
    rcpp_result_gen = Rcpp::wrap(rgwish_c(G, Ts, b, p, threshold_c));
    return rcpp_result_gen;
END_RCPP
}
// createDfName
Rcpp::StringVector createDfName(unsigned int D);
RcppExport SEXP _graphml_createDfName(SEXP DSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type D(DSEXP);
    rcpp_result_gen = Rcpp::wrap(createDfName(D));
    return rcpp_result_gen;
END_RCPP
}
// mat2df
Rcpp::DataFrame mat2df(arma::mat x, Rcpp::StringVector nameVec);
RcppExport SEXP _graphml_mat2df(SEXP xSEXP, SEXP nameVecSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type nameVec(nameVecSEXP);
    rcpp_result_gen = Rcpp::wrap(mat2df(x, nameVec));
    return rcpp_result_gen;
END_RCPP
}
// fitTree
Rcpp::List fitTree(Rcpp::DataFrame x, Rcpp::Formula formula);
RcppExport SEXP _graphml_fitTree(SEXP xSEXP, SEXP formulaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type x(xSEXP);
    Rcpp::traits::input_parameter< Rcpp::Formula >::type formula(formulaSEXP);
    rcpp_result_gen = Rcpp::wrap(fitTree(x, formula));
    return rcpp_result_gen;
END_RCPP
}
// getPartition
Rcpp::List getPartition(Rcpp::List tree, arma::mat supp);
RcppExport SEXP _graphml_getPartition(SEXP treeSEXP, SEXP suppSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type tree(treeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type supp(suppSEXP);
    rcpp_result_gen = Rcpp::wrap(getPartition(tree, supp));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_graphml_ep_logz", (DL_FUNC) &_graphml_ep_logz, 4},
    {"_graphml_getJT", (DL_FUNC) &_graphml_getJT, 1},
    {"_graphml_log_exp_mc", (DL_FUNC) &_graphml_log_exp_mc, 7},
    {"_graphml_gnorm_c", (DL_FUNC) &_graphml_gnorm_c, 4},
    {"_graphml_log_multi_gamma", (DL_FUNC) &_graphml_log_multi_gamma, 2},
    {"_graphml_log_wishart_norm", (DL_FUNC) &_graphml_log_wishart_norm, 3},
    {"_graphml_gnormJT", (DL_FUNC) &_graphml_gnormJT, 5},
    {"_graphml_lse", (DL_FUNC) &_graphml_lse, 2},
    {"_graphml_matrix2vector", (DL_FUNC) &_graphml_matrix2vector, 2},
    {"_graphml_getFreeElem", (DL_FUNC) &_graphml_getFreeElem, 2},
    {"_graphml_getNonFreeElem", (DL_FUNC) &_graphml_getNonFreeElem, 3},
    {"_graphml_init_graph", (DL_FUNC) &_graphml_init_graph, 3},
    {"_graphml_evalPsi", (DL_FUNC) &_graphml_evalPsi, 2},
    {"_graphml_approx_v1", (DL_FUNC) &_graphml_approx_v1, 4},
    {"_graphml_generalApprox", (DL_FUNC) &_graphml_generalApprox, 4},
    {"_graphml_approxWrapper", (DL_FUNC) &_graphml_approxWrapper, 7},
    {"_graphml_calcMode", (DL_FUNC) &_graphml_calcMode, 2},
    {"_graphml_approx_integral", (DL_FUNC) &_graphml_approx_integral, 4},
    {"_graphml_create_psi_mat_cpp", (DL_FUNC) &_graphml_create_psi_mat_cpp, 2},
    {"_graphml_psi_cpp", (DL_FUNC) &_graphml_psi_cpp, 2},
    {"_graphml_psi_cpp_mat", (DL_FUNC) &_graphml_psi_cpp_mat, 2},
    {"_graphml_grad_gwish", (DL_FUNC) &_graphml_grad_gwish, 2},
    {"_graphml_hess_gwish", (DL_FUNC) &_graphml_hess_gwish, 2},
    {"_graphml_support", (DL_FUNC) &_graphml_support, 2},
    {"_graphml_findAllCandidatePoints", (DL_FUNC) &_graphml_findAllCandidatePoints, 4},
    {"_graphml_createPartitionMap", (DL_FUNC) &_graphml_createPartitionMap, 2},
    {"_graphml_rcpparma_hello_world", (DL_FUNC) &_graphml_rcpparma_hello_world, 0},
    {"_graphml_rcpparma_outerproduct", (DL_FUNC) &_graphml_rcpparma_outerproduct, 1},
    {"_graphml_rcpparma_innerproduct", (DL_FUNC) &_graphml_rcpparma_innerproduct, 1},
    {"_graphml_rcpparma_bothproducts", (DL_FUNC) &_graphml_rcpparma_bothproducts, 1},
    {"_graphml_rgw", (DL_FUNC) &_graphml_rgw, 2},
    {"_graphml_rwish_c", (DL_FUNC) &_graphml_rwish_c, 3},
    {"_graphml_rgwish_c", (DL_FUNC) &_graphml_rgwish_c, 5},
    {"_graphml_createDfName", (DL_FUNC) &_graphml_createDfName, 1},
    {"_graphml_mat2df", (DL_FUNC) &_graphml_mat2df, 2},
    {"_graphml_fitTree", (DL_FUNC) &_graphml_fitTree, 2},
    {"_graphml_getPartition", (DL_FUNC) &_graphml_getPartition, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_graphml(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
