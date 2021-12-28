#ifndef UTIL_H
#define UTIL_H

double sse(arma::vec x, int n, double xbar);


/* sse(): 
   compute sum of squared error of input vector
*/ 
// [[Rcpp::export]]
double sse(arma::vec x, int n, double xbar) {
    double res = 0;
    for (u_int i = 0; i < n; i++) {
        res += std::pow(x(i)- xbar, 2);
    }
    return res;
} // end sse() function


#endif