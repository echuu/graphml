
#include "TruncParams.h"
#include "RcppArmadillo.h"

TruncParams::TruncParams(arma::vec logz_hat, arma::vec mu_hat, 
    arma::vec sigma_hat) {

    this->logz_hat = logz_hat;
    this->mu_hat = mu_hat;
    this->sigma_hat = sigma_hat;
}
arma::vec TruncParams::getlogzhat() {
    return this->logz_hat;
}
arma::vec TruncParams::getmuhat() {
    return this->mu_hat;
}
arma::vec TruncParams::getsigmahat() {
    return this->sigma_hat;
}
