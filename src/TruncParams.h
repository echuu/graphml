
#ifndef TRUNCPARAMS_H
#define TRUNCPARAMS_H

#include "RcppArmadillo.h"
#include <vector> 


class TruncParams {

    private: 
        double k;

    public: 
        arma::vec logz_hat;
        arma::vec mu_hat;
        arma::vec sigma_hat;
        TruncParams(arma::vec logz_hat, arma::vec mu_hat, arma::vec sigma_hat);
        arma::vec getlogzhat();
        arma::vec getmuhat();
        arma::vec getsigmahat();
};

#endif
