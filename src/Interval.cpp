

#include "Interval.h"

Interval::Interval(double lb, double ub, unsigned int feature) {
    this->lb = lb;
    this->ub = ub;
    this->feature = feature;
}

void Interval::print() {
        Rcpp::Rcout << "[ " << this->lb << ", " << 
        this->ub << "]" << std::endl;
}