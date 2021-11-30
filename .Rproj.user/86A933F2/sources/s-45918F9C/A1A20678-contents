

source("examples/gwish/gwish_density.R")
library(BDgraph)
#### initialize graphs ---------------------------------------------------------

p1 = 5
G_5 = matrix(c(1,1,0,1,1,
               1,1,1,0,0,
               0,1,1,1,1,
               1,0,1,1,1,
               1,0,1,1,1), p1, p1)

p2 = 9
G_9 = matrix(c(1,1,0,0,1,0,0,0,0,
               1,1,1,1,1,0,0,0,0,
               0,1,1,1,0,0,0,0,0,
               0,1,1,1,1,1,1,0,0,
               1,1,0,1,1,1,0,0,0,
               0,0,0,1,1,1,1,1,1,
               0,0,0,1,0,1,1,1,1,
               0,0,0,0,0,1,1,1,1,
               0,0,0,0,0,1,1,1,1), p2, p2)
a = c(1, 3, 2, 5, 4, 6, 7, 8, 9)
G_9 = G_9[a, a]

# p = p1 + p2
# G = matrix(0, p, p)
# G[1:p1, 1:p1] = G_5
# G[(p1+1):p, (p1+1):p] = G_9

## make massive graph
# n_decomp = 1
# # p2 = n_decomp * 9
# p = p1 + n_decomp * 9
# G_0 = diag(1, n_decomp) %x% G_9

n_G0 = 1
n_G1 = 0
G0 = diag(1, n_G0) %x% G_5
G1 = diag(1, n_G1) %x% G_9

G0_dim = nrow(G0)
G1_dim = nrow(G1)

p = G0_dim + G1_dim


G = matrix(0, p, p)
G[1:G0_dim, 1:G0_dim] = G0
G[(G0_dim+1):p, (G0_dim+1):p] = G1

b = 100
n = 100
V_5 = n * diag(1, G0_dim)
V_9 = n * diag(1, G1_dim)
V   = n * diag(1, p)


#### compute normalizing constant: I_{G_5} * I_{G_9} ---------------------------

I_G = function(delta) {
  7/2 * log(pi) + lgamma(delta + 5/2) - lgamma(delta + 3) +
    lgamma(delta + 1) + lgamma(delta + 3/2) + 2 * lgamma(delta + 2) +
    lgamma(delta + 5/2)
}
(Z_5  = log(2^(0.5*p1*b + 7)) + I_G(0.5*(b-2)) + (-0.5 * p1 * b - 7) * log(n))

Y = matrix(0, p2, p2)
S = matrix(0, p2, p2)
(Z_9 = -logHIWnorm(G_9, b, V_9))
# BDgraph::gnorm(G_9, b, V_9, 1000)

(Z = n_G0 * Z_5 + n_G1 * Z_9)
gnorm(G, b, V, 1000)

## compare to the normalizing constant in g-wishart paper
# I_G_9 = function(delta, D) {
#   p = nrow(D)
#   -(delta + 0.5 * (p + 1)) * log_det(D) + logmultigamma(p, delta + 0.5 * (p + 1))
# }
#
# lmg = function(p, alpha) {
#   p * (p - 1) / 4 * log(pi) + sum(lgamma(alpha - 0.5 * (1:p - 1)))
# }
#
# nEdge_G9 = sum(upper.tri(G_9) & G_9)
#
# (Z_9 = log(2^(0.5*p2*b + nEdge_G9)) + I_G_9(0.5 * (b-2), V_9))
# (Z_9 = log(2^(0.5*p2*b + nEdge_G9)) + I_G_9(0.5 * (b-2), V_9))
# gnorm(G_9, b, V_9, J)

# Z = Z_5 + Z_9 # normalizing constant for G
# Z
# ------------------------------------------------------------------------------
J = 200
gnorm(G, b, V, J) # gnorm estimate of the entire (appended graph)
Z

library(dplyr)

## try computing the normalizing constant of G_9 first as sanity check
FREE_PARAMS_ALL = c(upper.tri(diag(1, p), diag = T) & G)

edgeInd = G[upper.tri(G, diag = TRUE)] %>% as.logical

## construct A matrix so that we can compute k_i
A = (upper.tri(diag(1, p), diag = F) & G) + 0

k_i  = colSums(A) # see step 2, p. 329 of Atay
nu_i = rowSums(A) # see step 2, p. 329 of Atay
b_i = nu_i + k_i + 1
b_i

set.seed(1)
Omega_G = rgwish(1, G, b, V) # generate the true precision matrix
P = chol(solve(V)) # upper cholesky factor; D^(-1) = TT'  in Atay paper

# params = list(G = G, P = P, p = p, edgeInd = edgeInd,
#               b = b, nu_i = nu_i, b_i = b_i)
N = 0
S = matrix(0, p, p)
D = sum(edgeInd) # number of free parameters / dimension of parameter space

index_mat = matrix(0, p, p)
index_mat[upper.tri(index_mat, diag = T)][edgeInd] = 1:D
index_mat[upper.tri(index_mat, diag = T)]
t_ind = which(index_mat!=0,arr.ind = T)
t_ind

index_mat[lower.tri(index_mat)] = NA
vbar = which(index_mat==0,arr.ind = T) # non-free elements
n_nonfree = nrow(vbar)

params = list(G = G, P = P, p = p, D = D, edgeInd = edgeInd,
              b = b, nu_i = nu_i, b_i = b_i,
              t_ind = t_ind, n_nonfree = n_nonfree, vbar = vbar)


samps = samplegw(J, G, b, N, V, S, solve(P), FREE_PARAMS_ALL)
u_samps = samps$Psi_free %>% data.frame

u_df = preprocess(u_samps, D, params)     # J x (D_u + 1)
u_df %>% head

grad = function(u, params) { pracma::grad(psi, u, params = params) }
hess = function(u, params) { pracma::hessian(psi, u, params = params) }
u_star = globalMode(u_df)
u_star_numer = u_star

grad = function(u, params) { f(u, params)  }
hess = function(u, params) { ff(u, params) }
u_star = globalMode(u_df)
u_star_closed = u_star

cbind(u_star_numer, u_star_closed)

samps = samplegw(J, G, b, N, V, S, solve(P), FREE_PARAMS_ALL)
u_samps = samps$Psi_free %>% data.frame
u_df = preprocess(u_samps, D, params)     # J x (D_u + 1)
logzhat = hybridml::hybml(u_df, params, psi = psi, grad = grad, hess = hess, u_0 = u_star)$logz
logzhat           # hybrid
Z                 # truth
gnorm(G, b, V, J) # gnorm estimate of the entire (appended graph)



# ------------------------------------------------------------------------------

log_density = function(u, data) {
  -psi(u, data)
}
J = 1000
samps = samplegw(J, G, b, N, V, S, solve(P), FREE_PARAMS_ALL)
u_samps = samps$Psi_free %>% data.frame
u_samp = as.matrix(u_samps)
colnames(u_samp) = names(u_df)[1:D]
# prepare bridge_sampler input()
lb = rep(-Inf, D)
ub = rep(Inf, D)
names(lb) <- names(ub) <- colnames(u_samp)

bridge_result = bridgesampling::bridge_sampler(samples = u_samp,
                                               log_posterior = log_density,
                                               data = params,
                                               lb = lb, ub = ub,
                                               method = 'normal',
                                               silent = TRUE)
bridge_result$logml
abs(Z - bridge_result$logml)
Z

truth = Z

abs(Z - bridge_result$logml)
abs(Z - logzhat)

n_sims = 50
hyb = numeric(n_sims)
gnorm_approx = numeric(n_sims)
# bridge = numeric(n_sims)
j = 1
set.seed(1)
truth = Z
while (j <= n_sims) {

  samps = samplegw(J, G, b, N, V, S, solve(P), FREE_PARAMS_ALL)
  u_samps = samps$Psi_free %>% data.frame
  u_df = preprocess(u_samps, D, params)     # J x (D_u + 1)

  hyb[j] = hybridml::hybml(u_df, params, psi = psi, grad = grad,
                           hess = hess, u_0 = u_star)$logz

  ### bridge estimator ---------------------------------------------------------
  # u_samp = as.matrix(u_samps)
  # colnames(u_samp) = names(u_df)[1:D]
  # # prepare bridge_sampler input()
  # lb = rep(-Inf, D)
  # ub = rep(Inf, D)
  # names(lb) <- names(ub) <- colnames(u_samp)
  #
  # bridge_result = bridgesampling::bridge_sampler(samples = u_samp,
  #                                                log_posterior = log_density,
  #                                                data = params,
  #                                                lb = lb, ub = ub,
  #                                                silent = TRUE)
  # bridge_result$logml
  # bridge[j] = bridge_result$logml
  ### bridge estimator ---------------------------------------------------------

  # gnorm_approx[j] = gnorm(G, b, V, J)

  print(paste('iter ', j, ': ',
              round(mean(hyb[hyb!=0]), 3),
              ' (error = ',
              round(mean(abs(truth - hyb[hyb!=0])), 3), ')',
              sep = ''))

  # print(paste('iter ', j, ': ',
  #             round(mean(bridge[bridge!=0]), 3),
  #             ' (error = ',
  #             round(mean(abs(truth - bridge[bridge!=0])), 3), ')',
  #             sep = ''))

  # print(paste('iter ', j, ': ',
  #             round(mean(gnorm_approx[gnorm_approx!=0]), 3),
  #             ' (error = ',
  #             round(mean(truth - gnorm_approx[gnorm_approx!=0]), 3),
  #             ')', sep = ''))

  j = j + 1
}





gnorm_approx = gnorm_approx[1:50]
bridge = bridge[1:50]

approx = data.frame(truth, hyb = hyb, gnorm = gnorm_approx, bridge = bridge)
data.frame(logz = colMeans(approx), approx_sd = apply(approx, 2, sd),
           avg_error = colMeans(truth - approx))



approx = data.frame(truth, hyb = hyb, gnorm = gnorm_approx, bridge = bridge)
data.frame(logz = colMeans(approx), approx_sd = apply(approx, 2, sd),
           abs_err = colMeans((truth - approx)),
           rmse = sqrt(colMeans((truth - approx)^2)))


# ------------------------------------------------------------------------------

























## try computing the normalizing constant of G_9 first as sanity check
FREE_PARAMS_ALL = c(upper.tri(diag(1, p2), diag = T) & G_9)

edgeInd = G_9[upper.tri(G_9, diag = TRUE)] %>% as.logical

## construct A matrix so that we can compute k_i
A = (upper.tri(diag(1, p2), diag = F) & G_9) + 0

k_i  = colSums(A) # see step 2, p. 329 of Atay
nu_i = rowSums(A) # see step 2, p. 329 of Atay
b_i = nu_i + k_i + 1
b_i


p = p2
set.seed(1)
Omega_G = rgwish(1, G_9, b, V_9) # generate the true precision matrix

P = chol(solve(V_9)) # upper cholesky factor; D^(-1) = TT'  in Atay paper

params = list(G = G_9, P = P, p = p, edgeInd = edgeInd,
              b = b, nu_i = nu_i, b_i = b_i)
N = 0
S = matrix(0, p, p)
D = sum(edgeInd) # number of free parameters / dimension of parameter space

J = 2000
samps = samplegw(J, G_9, b, N, V_9, S, solve(P), FREE_PARAMS_ALL)
u_samps = samps$Psi_free %>% data.frame

## check equality of Psi -- uncomment print(u_mat) line in psi() function
matrix(samps$Psi[1,], p, p)
u = unlist(unname(u_samps[1,]))
test = psi(u, params)

all.equal(test, matrix(samps$Psi[1,], p, p))

u_df = preprocess(u_samps, D, params)     # J x (D_u + 1)
u_df %>% head

grad = function(u, params) { pracma::grad(psi, u, params = params) }
hess = function(u, params) { pracma::hessian(psi, u, params = params) }
u_star = globalMode(u_df)
u_star

### (1) find global mean
# MAP_LOC = which(u_df$psi_u == min(u_df$psi_u))
# u_0 = u_df[MAP_LOC,1:D] %>% unname() %>% unlist()
# u_star = u_0
gnorm(G_9, b, V_9, J)
logzhat = hybridml::hybml(u_df, params, psi = psi, grad = grad, hess = hess, u_0 = u_star)$logz
logzhat
Z_9















