
#ifndef INTERVAL_H
#define INTERVAL_H

#include "graphml_types.h"
#include <vector>


struct Interval {
    public:
        double lb;
        double ub;
        unsigned int feature;
        Interval(double lb, double ub, unsigned int feature) {
            this->lb = lb;
            this->ub = ub;
            this->feature = feature;
        }
        void print() {
             Rcpp::Rcout << "[ " << this->lb << ", " << 
                this->ub << "]" << std::endl;
        }
};






#endif
