
init_graph_post = function(G, b, n, V, Y) {

  # input:
  # G: graph in adjacency matrix form
  # b: degrees of freedom for g-wishart distribution
  # n: pseudo-sample size, used to initialize the scale matrix
  # V: unscaled matrix
  #
  # output: list with various quantities that will be used in future calcs
  # G:         graph
  # b:         degrees of freedom
  # n:         pseudo-sample size, used to initialize the scale matrix
  # p:         dimension of graph
  # V_n:       unscaled matrix
  # S:         S = X'X when there is data (for posteriors)
  # P:         upper cholesky factor of the scale matrix V_n
  # D:         dimension of parameter space (# of free parameters)
  # edgeInd:   bool indicator for edges in G (upper tri + diag elements only)
  # nu_i       defined in step 2, p. 329 of Atay
  # b_i:       defined in step 2, p. 329 of Atay
  # t_ind:     (row,col) of the location of each of the free parameters
  # n_nonfree: number of nonfree parameters
  # vbar:      nonfree elements

  p   = ncol(G)           # dimension fo the graph

  ###### LINE BELOW IS CHANGED TO ACCOMMODATE POSTERIOR DISTRIBUTIONS ##########
  # V_n = n * V             # scale matrix for gW distribution
  # S   = matrix(0, p, p)   # S = X'X when there is data (for posteriors)
  S   = t(Y)%*% Y
  V_n = S + V  # posterior scale matrix
  b_n = b + n  # posterior degrees of freedom
  ##############################################################################
  P   = chol(solve(V_n))  # upper cholesky factor; D^(-1) = TT' in Atay paper


  FREE_PARAMS_ALL = c(upper.tri(diag(1, p), diag = T) & G)
  edgeInd = G[upper.tri(G, diag = TRUE)] %>% as.logical

  ## construct A matrix so that we can compute k_i
  A = (upper.tri(diag(1, p), diag = F) & G) + 0

  k_i  = colSums(A) # see step 2, p. 329 of Atay
  nu_i = rowSums(A) # see step 2, p. 329 of Atay
  b_i = nu_i + k_i + 1

  D = sum(edgeInd) # number of free parameters / dimension of parameter space

  index_mat = matrix(0, p, p)
  index_mat[upper.tri(index_mat, diag = T)][edgeInd] = 1:D
  index_mat[upper.tri(index_mat, diag = T)]
  t_ind = which(index_mat!=0,arr.ind = T)

  index_mat[lower.tri(index_mat)] = NA
  vbar = which(index_mat==0,arr.ind = T) # non-free elements
  n_nonfree = nrow(vbar)



  obj = list(G = G, b = b, n = n, p = p, V_n = V_n, b_n = b_n,
             S = S, P = P, P_inv = solve(P),
             D = D, edgeInd = edgeInd, FREE_PARAMS_ALL = FREE_PARAMS_ALL,
             nu_i = nu_i, b_i = b_i,
             t_ind = t_ind, n_nonfree = n_nonfree, vbar = vbar,
             n_graphs = 1)

  return(obj)
}


samplegw_post = function(J, G, b_n, V_n, P_inv, param_ind) {

  ## sample from gw distribution: stored as J - (D x D) matrices
  Omega_post = rgwish(J, G, b_n, V_n)

  ## Compute Phi (upper triangular), stored column-wise, Phi'Phi = Omega
  Phi = apply(Omega_post, 3, chol) # (p^2) x J

  ## Compute Psi
  Psi = apply(Phi, 2, computePsi, P_inv = P_inv)

  ## Extract free elements of Psi
  Psi_free = apply(Psi, 2, extractFree, free_ind = param_ind)

  out = list(Phi = t(Phi),
             Psi = t(Psi),
             Psi_free = t(Psi_free),
             Omega = Omega_post)

}


psi_post = function(u, params, DEBUG = FALSE) {

  p     = params$p
  n     = params$n
  P     = params$P
  b     = params$b
  b_i   = params$b_i
  nu_i  = params$nu_i
  P_inv = params$P_inv
  S     = params$S
  Psi   = create_psi_mat(u, params)
  Phi   = Psi %*% P
  Omega = t(Phi) %*% Phi

  # compute log likelihood
  loglik = - 0.5 * n * p * log(2 * pi) + 0.5 * n * log_det(Omega) -
    0.5 * matrixcalc::matrix.trace(Omega %*% S)

  # compute log prior
  t0 = p * log(2) +
    sum((b + b_i - 1) * log(diag(P)) + (b + nu_i - 1) * log(diag(Psi)))
  t1 = -0.5 * sum(Psi[upper.tri(Psi, diag = TRUE)]^2)
  logprior = t0 + t1

  if (DEBUG) {
    print("psi:"); print(Psi); cat("\n")
    print("phi:"); print(Phi); cat("\n")
    print("omega:"); print(Omega); cat("\n")
    print("loglik:"); print(loglik); cat("\n")
    print("logprior:"); print(logprior); cat("\n")
  }

  # return negative log posterior
  return(-logprior-loglik)
}

preprocess_post = function(post_samps, D, params) {
    # (J x 1)
    psi_u = apply(post_samps, 1, psi_post, params = params) %>% unname()

    # (1.2) name columns so that values can be extracted by partition.R
    u_df_names = character(D + 1)
    for (d in 1:D) {
      u_df_names[d] = paste("u", d, sep = '')
    }
    u_df_names[D + 1] = "psi_u"

    # populate u_df
    u_df = cbind(post_samps, psi_u) # J x (D + 1)
    names(u_df) = u_df_names

    return(u_df)
}

gwish_globalMode_post = function(u_df, params,
                                 psi, grad, hess,
                                 tolerance = 1e-5, maxsteps = 200) {

  # use the MAP as the starting point for the algorithm
  MAP_LOC = which(u_df$psi_u == min(u_df$psi_u))
  theta = u_df[MAP_LOC,1:params$D] %>% unname() %>% unlist()

  numsteps = 0
  tolcriterion = 100
  step.size = 1

  while(tolcriterion > tolerance && numsteps < maxsteps){
    # print(numsteps)
    # hess_obj = hess(theta, params_G5)
    G = -hess(theta, params)
    invG = solve(G)
    # G = -hess(theta, params)
    # invG = solve(G)

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

psi_post(u1, G5_obj)
u_df_post = preprocess_post(u_samps, G5_obj$D, G5_obj)



#### start problem

## initialize the graph, posterior parameters
set.seed(2021)
p = 4; n = 100
G = matrix(0, p, p)
b = 3
G[1, p] = 1
for(i in 1:(p-1)) { G[i, i+1] = 1 }
G = G + t(G); diag(G) = 1
Sigma = solve(rgwish(1, G, b, diag(p)))
Y = matrix(rnorm(n*p,), n, p)
Y = Y %*% t(chol(Sigma)) # (n x p)
V = diag(1, p)

## initialize graph object that will be used with the hybrid algorithm
GG = init_graph_post(G = G, b = b, n = n, V = V, Y = Y)


## obtain posterior samples
J = 2000
samps = samplegw_post(J, GG$G, GG$b_n, GG$V_n, solve(GG$P), GG$FREE_PARAMS_ALL)
u_samps = samps$Psi_free %>% data.frame
## evaluate negative log posterior
u_df_post = preprocess_post(u_samps, GG$D, GG)


### testing log likelihood calculation -----------------------------------------
u1 = unname(unlist(u_df_post[1,1:GG$D]))

- 0.5 * n * p * log(2 * pi) + 0.5 * n * log_det(samps$Omega[,,1]) -
  0.5 * matrixcalc::matrix.trace(samps$Omega[,,1] %*% (t(Y) %*% Y))

psi_post(u1, GG, DEBUG = TRUE)

psi(u1, GG)
pracma::grad(psi, u1, params = GG)
pracma::hessian(psi, u1, params = GG)




### ----------------------------------------------------------------------------

## define gradient and hessian functions
grad_post = function(u, params) {
  pracma::grad(psi_post, u, params = params)
}
hess_post = function(u, params) {
  pracma::hessian(psi_post, u, params = params)
}

## compute posterior mode
u_star = gwish_globalMode_post(u_df_post, GG,
                               psi = psi_post, grad = grad_post, hess = hess_post)

## compute approximation to log marginal likelihood
logz = hybml(u_df = u_df_post, params = GG, u_0 = u_star,
             psi = psi_post, grad = grad_post, hess = hess_post)$logz
logz

(gnorm_approx = - 0.5 * p * n * log(2 * pi) +
  gnorm(G, n + b, V + t(Y) %*% Y, 1000) - gnorm(G, b, V, 1000))


logz - gnorm_approx

library(bridgesampling)
log_density = function(u, data) {
  -psi_post(u, data)
}

u_samp = as.matrix(u_samps)
colnames(u_samp) = names(u_df_post)[1:GG$D]
# prepare bridge_sampler input()
lb = rep(-Inf, GG$D)
ub = rep(Inf, GG$D)
names(lb) <- names(ub) <- colnames(u_samp)

bridge_result = bridgesampling::bridge_sampler(samples = u_samp,
                                               log_posterior = log_density,
                                               data = GG,
                                               lb = lb, ub = ub,
                                               method = 'warp3',
                                               silent = TRUE)
bridge_result$logml


set.seed(2021)
p = 30; n = 300
G = matrix(0, p, p)
b = 3
G[1, p] = 1
for(i in 1:(p-1)){
  G[i, i+1] = 1
}

G = G + t(G); diag(G) = 1
Sigma = rgwish(1, G, b, diag(p))
Y = matrix(rnorm(n*p,), n, p)
Y = Y %*% t(chol(Sigma))
gnorm(G, n+b, diag(p)+t(Y)%*%Y, 1000)
















#### random test stuff

## check psi value
create_psi_mat(u1, GG)

## check phi value
matrix(samps$Phi[1,], 5, 5)

## check omega value
samps$Omega[,,1]

psi(u1, GG)
pracma::grad(psi, u1, params = GG)
pracma::hessian(psi, u1, params = GG)

psi_cpp(u1, G5_obj)
grad_cpp(u1, G5_obj)
hess_cpp(u1, G5_obj)












