
## gwish_uhler.R
## author: Eric Chuu

## This file compares the performance of the GNORM, EPSOM-HybJT estimators
## for high-dimensional graphs that are constructed using G_5, as described
## in Section 4.3.1 of the paper. We evaluate these estimates using the
## the exact normalizing constant which can calculated with the formula
## from Uhler et al. (2016).



library(graphml) # load package for EPSOM-HybJT estimator
library(BDgraph) # load package for GNORM estimator
library(dplyr)   # load package for data manipulation
library(Deriv)   # load package for numerical derivative for exact GW calc

set.seed(1)
p1 = 5

# initialize G_5 as given in the paper
G_5 = matrix(c(1,1,0,1,1,
               1,1,1,0,0,
               0,1,1,1,1,
               1,0,1,1,1,
               1,0,1,1,1), p1, p1)

n_edge_G5 = 7 # number of edges in G_5

# initialize the GW parameters, GW(delta, V)
b = 100                                   # degrees of freedom
delta = 0.5*(b-2)                         # alt. form of degrees of freedom
V = BDgraph::rgwish(1, G_5, b, diag(p1))
V = cov2cor(V)                            # scale matrix
J = 1000                                  # number of MCMC samples

#### -- compute the exact normalizing constant as given in Uhler et. al. -- ####

## (1) compute constants
V_13 = V[c(1,3), c(1,3)]
c1 = (V_13[1,1] * V_13[2,2] - V_13[1,2] * V_13[2,1])^(2 * delta + 9/2) /
  ( det(V[c(1:3), c(1:3)])^(delta + 2) *
      det(V[c(1,3:5), c(1,3:5)])^(delta + 5/2) )

V_1345 = V[c(1,3), c(4,5)]
det_V_1345 = V_1345[1,1] * V_1345[2,2] - V_1345[1,2] * V_1345[2,1]

V_45 = V[c(4,5), c(4,5)]
tmp_numer = V_45 - t(V_1345) %*% solve(V_13) %*% V_1345
det_numer = tmp_numer[1,1] * tmp_numer[2,2] - tmp_numer[1,2] * tmp_numer[2,2]
final_numer = det_numer^(delta + 3/2)

det_denom = V_45[1,1] * V_45[2,2] - V_45[1,2] * V_45[2,1]

X = V[c(4,5), c(4,5)]
Y = V[1, c(4,5)]
Z = V[2, c(4,5)]

I_G = function(delta) {
  7/2 * log(pi) + lgamma(delta + 5/2) - lgamma(delta + 3) +
    lgamma(delta + 1) + lgamma(delta + 3/2) + 2 * lgamma(delta + 2) +
    lgamma(delta + 5/2)
}

IG5_I5 = I_G(delta)

s = function(t1, t2, q) {
  tmp = X[1,1] * X[2,2] - t1 * X[1,1] * Y[2]^2 - t2 * X[1,1] * Z[2]^2 -
    t1 * Y[1]^2 * X[2,2] + t1^2 * Y[1]^2 * Y[2]^2 + t1 * t2 * Y[1]^2 * Z[2]^2 -
    t2 * Z[1]^2 * X[2,2] + t1 * t2 * Z[1]^2 * Y[2]^2 + t2^2 * Z[1]^2 * Z[2]^2 -
    X[1,2]^2 + 2 * t1 * X[1,2] * Y[1] * Y[2] + 2 * t2 * X[1,2] * Z[1] * Z[2] -
    2 * t1 * t2 * Y[1] * Y[2] * Z[1] * Z[2] - t1^2 * Y[1]^2 * Y[2]^2 -
    t2^2 * Z[1]^2 * Z[2]^2
  tmp^(-delta-q-3/2)
}

rf = function(a, k) {
  gamma(a + k) / gamma(a)
} # end rf() function

# precompute the derivatives required in the normalizing consant formula
d1s = Deriv(s, x = c("t1", "t2"), nderiv = 1)
d2s = Deriv(s, x = c("t1", "t2"), nderiv = 2)
d3s = Deriv(s, x = c("t1", "t2"), nderiv = 3)
d4s = Deriv(s, x = c("t1", "t2"), nderiv = 4)
d5s = Deriv(s, x = c("t1", "t2"), nderiv = 5)


## calculate the summation given in Eq. (4.1) in Uhler et. al. (2016)
gg = function(l_max) {
  q_sum = 0
  q_max = 10
  for (q in 0:q_max) {

    q_tmp = rf(delta + 5/2, q) * rf(delta + 3/2, q) /
      (factorial(q) * rf(delta + 3, 2*q)) * det_V_1345^(2*q)

    ## evaluate derivatives at (0,0) with q = q
    deriv_list = list(
      d1s_00 = d1s(0, 0, q = q),
      d2s_00 = d2s(0, 0, q = q),
      d3s_00 = d3s(0, 0, q = q),
      d4s_00 = d4s(0, 0, q = q),
      d5s_00 = d5s(0, 0, q = q)
    )

    l_sum = 0
    for (l in 0:l_max) {

      l_tmp = 1 / (factorial(l) * rf(delta + 2*q + 3, l))
      ### compute term for l = 0
      l0 = final_numer / det_denom^(delta + q + 3/2)

      l_vec = numeric(l+1)
      if (l == 0) {
        l_vec[l+1] = l0
      } else {
        l_vec = final_numer * deriv_list[[l]][2^(0:l)]
      }
      l12_sum = sum(
        choose(l, 0:l) * rf(delta + q + 5/2, 0:l) *
          rf(delta + q + 5/2, l:0) * l_vec
      )
      l_sum = l_sum + l_tmp * l12_sum

    } # end summation over l
    q_sum = q_sum + q_tmp * l_sum
  } # end summation over q
  unname(IG5_I5 + log(c1) + log(q_sum))
} # end gg() function

#### ------------ end of exact normalizing constant calculation ----------- ####


#### compare normalizing constant estimates with the exact value

const_offset_G5 = (0.5 * p1 * b + n_edge_G5) * log(2)

# exact GW(b, V) log normalizing constant
gg(5) + const_offset_G5

# EPSOM-HybJT estimate of GW(b, V) log normalizing constant
graphml::hybridJT(G_5, getEdgeMat(G_5), b, V, J)

# GNORM estimate of GW(b, V)  log normalizing constant
BDgraph::gnorm(G_5, b, V, J)


#### consider the high-dimensional case where we stack G_5 along a block diag

n_graphs = 10                  # number of times G_5 is stacked
G = diag(1, n_graphs) %x% G_5  # (n_graphs * p1) x (n_graphs * p1)
p = ncol(G)                    # p = n_graphs * p1
D = diag(1, n_graphs) %x% V    # (n_graphs * p1) x (n_graphs * p1)
n_edge = n_graphs * n_edge_G5

n_params = sum(G[upper.tri(G, diag = TRUE)] %>% as.logical)

# compute the constant that we add to Uhler's formula
const_offset = (0.5 * p * b + n_edge) * log(2)

# compute the exact formula for the stacked graph
(truth = n_graphs * gg(5) + const_offset)

# EPSOM-HybJT estimate of GW(b, D) log normalizing constant
BDgraph::gnorm(G, b, D, J)

# GNORM estimate of GW(b, V)  log normalizing constant
graphml::hybridJT(G, getEdgeMat(G), b, D, J)
