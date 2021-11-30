
library(dplyr)
source("C:/Users/ericc/Documents/hybridml/examples/hiw/hiw_helper.R")

Rcpp::sourceCpp("C:/Users/ericc/mlike_approx/speedup/hiw.cpp")

library(dplyr)
testG  = matrix(c(1,1,0,0,0,
                  1,1,1,1,1,
                  0,1,1,1,1,
                  0,1,1,1,1,
                  0,1,1,1,1), 5, 5)


# Given graph testG ------------------------------------------------------------
testG = matrix(c(1,1,0,0,1,0,0,0,0,
                 1,1,1,1,1,0,0,0,0,
                 0,1,1,1,0,0,0,0,0,
                 0,1,1,1,1,1,1,0,0,
                 1,1,0,1,1,1,0,0,0,
                 0,0,0,1,1,1,1,1,1,
                 0,0,0,1,0,1,1,1,1,
                 0,0,0,0,0,1,1,1,1,
                 0,0,0,0,0,1,1,1,1), 9, 9)

a = c(1, 3, 2, 5, 4, 6, 7, 8, 9)
testG = testG[a, a]

# ------------------------------------------------------------------------------


D = nrow(testG)
b = 3          # prior degrees of freedom
V = diag(1, D) # prior scale matrix

D_0 = 0.5 * D * (D + 1) # num entries on diagonal and upper diagonal
J = 1000

# logical vector determining existence of edges between vertices
edgeInd = testG[upper.tri(testG, diag = TRUE)] %>% as.logical
upperInd = testG[upper.tri(testG)] %>% as.logical
D_u = sum(edgeInd)

# Specify true value of Sigma
set.seed(1)
true_params = HIWsim(testG, b, V)
Sigma_G = true_params$Sigma
Omega_G = true_params$Omega # precision matrix -- this is the one we work with

# chol(Omega_G)
# Generate data Y based on Sigma_G
N = 100
Y = matrix(0, N, D)
for (i in 1:N) {
  Y[i, ] = t(t(chol(Sigma_G)) %*% rnorm(D, 0, 1)) # (500 x D)
}

S = t(Y) %*% Y

nu = rowSums(chol(Omega_G) != 0) - 1
xi = b + nu - 1
t_ind = which(chol(Omega_G) != 0, arr.ind = T)


params = list(N = N, D = D, D_0 = D_0, D_u = D_u,
              testG = testG, edgeInd = edgeInd,
              upperInd = upperInd, S = S, V = V, b = b,
              nu = nu, xi = xi, G = G, t_ind = t_ind)

J = 1000
postIW = sampleHIW(J, D_u, D_0, testG, b, N, V, S, edgeInd)
post_samps = postIW$post_samps                 # (J x D_u)
u_df = hybridml::preprocess(post_samps, D_u, params)     # J x (D_u + 1)
u_df = preprocess(post_samps, D_u, params)     # J x (D_u + 1)

(LIL = logmarginal(Y, testG, b, V, S))

# slow_grad = function(u, params) { pracma::grad(old_psi, u, params = params) }
# slow_hess = function(u, params) { pracma::hessian(old_psi, u, params = params) }

globalMode = function(u_df, params, D = ncol(u_df) - 1, tolerance = 0.00001, maxsteps = 200) {

  # use the MAP as the starting point for the algorithm
  MAP_LOC = which(u_df$psi_u == min(u_df$psi_u))[1]
  theta = u_df[MAP_LOC,1:D] %>% unname() %>% unlist()

  numsteps = 0
  tolcriterion = 100
  step.size = 1


  while(tolcriterion > tolerance && numsteps < maxsteps){

    G = -hess(theta, params)
    G = prama::hessian(old_psi, theta, params = params)
    invG = solve(G)
    thetaNew = theta + step.size * invG %*% grad(theta, params)

    # if precision turns negative or if the posterior probability of
    # thetaNew becomes smaller than the posterior probability of theta
    if(-psi(thetaNew, params) < -psi(theta, params)) {
      cat('tolerance reached on log scale =', tolcriterion, '\n')
      print(paste("converged -- ", numsteps, " iters", sep = ''))
      return(theta)
    }

    tolcriterion = abs(psi(thetaNew, params)-psi(theta, params))
    theta = thetaNew
    numsteps = numsteps + 1
  }

  if(numsteps == maxsteps)
    warning('Maximum number of steps reached in Newton method.')

  print(paste("converged -- ", numsteps, " iters", sep = ''))
  return(theta)
}



u_star = globalMode(u_df, params)
out = hybridml::hybml(u_df, params, grad = grad, hess = hess, u_0 = u_star)
out = hybridml::hybml(u_df, params, grad = fast_grad, hess = fast_hess, u_0 = u_star)
out$logz

(LIL = logmarginal(Y, testG, b, V, S))

- 0.5 * D * N * log(2 * pi) + BDgraph::gnorm(testG, b + N, V + S, iter = 1000) -
  BDgraph::gnorm(testG, b, V, iter = 1000)

library(bridgesampling)
log_density = function(u, data) {
  -psi(u, data)
}


n_sims = 100
hyb = numeric(n_sims)
hyb_old = numeric(n_sims)
gnorm_approx = numeric(n_sims)
bridge = numeric(n_sims)
bridge_warp  = numeric(n_sims)

j = 1
set.seed(1)
truth = LIL
while (j <= n_sims) {

  postIW = sampleHIW(J, D_u, D_0, testG, b, N, V, S, edgeInd)
  post_samps = postIW$post_samps                 # (J x D_u)
  u_df = hybridml::preprocess(post_samps, D_u, params)     # J x (D_u + 1)

  hyb[j] = hybridml::hybml(u_df, params, grad = grad, hess = hess, u_0 = u_star)$logz
  hyb_old[j] = hybridml::hybml_const(u_df)$logz

  ### bridge estimator ---------------------------------------------------------
  u_samp = as.matrix(post_samps)
  colnames(u_samp) = names(u_df)[1:D_u]
  # prepare bridge_sampler input()
  lb = rep(-Inf, D_u)
  ub = rep(Inf, D_u)
  names(lb) <- names(ub) <- colnames(u_samp)

  bridge_result = bridgesampling::bridge_sampler(samples = u_samp,
                                                 log_posterior = log_density,
                                                 data = params,
                                                 lb = lb, ub = ub,
                                                 silent = TRUE)

  bridge[j] = bridge_result$logml

  bridge_result = bridgesampling::bridge_sampler(samples = u_samp,
                                                 log_posterior = log_density,
                                                 data = params,
                                                 lb = lb, ub = ub,
                                                 method = 'warp3',
                                                 silent = TRUE)
  bridge_warp[j] = bridge_result$logml
  ### bridge estimator ---------------------------------------------------------

  gnorm_approx[j] = - 0.5 * D * N * log(2 * pi) +
    BDgraph::gnorm(testG, b + N, V + S, iter = J/2) -
    BDgraph::gnorm(testG, b, V, iter = J/2)


  if (j %% 5 == 0) {
    print(paste('iter ', j, ': ', ' hyb: ',
                round(mean(hyb[hyb!=0]), 3),
                ' (error = ',
                round(mean(abs(truth - hyb[hyb!=0])), 3), ')',
                sep = ''))
  }

  # print(paste('iter ', j, ': ', ' hyb: ',
  #             round(mean(hyb[hyb!=0]), 3),
  #             ' (error = ',
  #             round(mean(abs(truth - hyb[hyb!=0])), 3), ')',
  #             sep = ''))
  #
  # print(paste('iter ', j, ': ', ' bdg: ',
  #             round(mean(bridge[bridge!=0]), 3),
  #             ' (error = ',
  #             round(mean(abs(truth - bridge[bridge!=0])), 3), ')',
  #             sep = ''))
  #
  # print(paste('iter ', j, ': ', ' gnm: ',
  #             round(mean(gnorm_approx[gnorm_approx!=0]), 3),
  #             ' (error = ',
  #             round(mean(truth - gnorm_approx[gnorm_approx!=0]), 3),
  #             ')', sep = ''))

  j = j + 1
}



approx = data.frame(truth, hyb = hyb, gnorm = gnorm_approx, bridge = bridge,
                    bridge_warp = bridge_warp)
approx_long = reshape2::melt(approx, id.vars = 'truth')

res_tbl =
  data.frame(logz      = colMeans(approx),
             approx_sd = apply(approx, 2, sd),
             avg_error = colMeans(truth - approx),            # avg error
             mae       = colMeans(abs(truth - approx)),       # mean absolute error
             rmse      = sqrt(colMeans((truth - approx)^2)))  # root mean square error

t(round(res_tbl, 4)[,-c(2,4)])[,c(1,4,5,3,2)]

## truth compared to:
  ## bridge, warped bridge
  ## hybep, hyb
  ## gnorm
  ## moulines?








