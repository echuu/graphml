
setwd("C:/Users/ericc/mlike_approx/algo")
source("setup.R")           # setup global environment, load in algo functions
## 

source("C:/Users/ericc/mlike_approx/bayes_regression/bayesLinRegHelper.R") 
# source("C:/Users/ericc/mlike_approx/paper_simulations/table2/mvn_estimators.R") 

# setwd("/mlike_approx/algo")
# source("setup.R")           # setup global environment, load in algo functions
# source("/mlike_approx/bayes_regression/bayesLinRegHelper.R") 
# source("/mlike_approx/paper_simulations/table2/mvn_estimators.R")


D = c(4) # test for smalller dimensions for now
N = c(100) # for testing -- comment this line to perform ext. analysis


## priors ----------------------------------------------------------------------
set.seed(1)
mu_0 = rep(0, D)      # prior mean for beta
tau  = 1 / 4          # precision: inverse of variance
sigmasq = 4           # true variance (1 x 1) 

## true beta -------------------------------------------------------------------

beta = sample(-10:10, D, replace = T) 

## simulate regression data ----------------------------------------------------

X   = matrix(rnorm(N * D), N, D)                # (N x D) design matrix
eps = rnorm(N, mean = 0, sd = sqrt(sigmasq))    # (N x 1) errors vector
y   = X %*% beta + eps                          # (N x 1) response vector

data = list(X = X, y = y)


## compute posterior parameters ------------------------------------------------

Q_beta = 1 / sigmasq * (t(X) %*% X + tau * diag(1, D))
Q_beta_inv = solve(Q_beta)
b = 1 / sigmasq * t(X) %*% y
mu_beta = Q_beta_inv %*% b

prior = list(y = y, X = X, sigmasq = sigmasq, tau = tau, N = N, D = D)
post = list(Q_beta = Q_beta, Q_beta_inv = Q_beta_inv, mu_beta = mu_beta, b = b)


## algorithm settings ----------------------------------------------------------
J         = 2000         # number of MC samples per approximation
# ------------------------------------------------------------------------------

## sample from posterior -------------------------------------------------------

# true log marginal likelihood
# lil(prior, post) 

u_samps = rmvnorm(J, mean = c(mu_beta), sigma = Q_beta_inv) %>% data.frame 
u_df = preprocess(u_samps, D, prior) # J x (D + 1) -- stored row-wise 

# library(microbenchmark)
# microbenchmark("hml" = hml_const(1, D, u_df, J, prior))
# start_time <- Sys.time()
hml_approx = hml_const(1, D, u_df, J, prior)
# end_time <- Sys.time()

# end_time - start_time
hml_approx$const_vec       # -272.1245

(LIL = lil(prior, post))   # -272.1202

hybrid_ml(D, u_df, J, prior)$zhat



n_samps = 10

# for the partition learned from prev fitted tree, extract the partition id and
# the optimal value of psi for this partition
og_part = hml_approx$param_out %>% 
    dplyr::select(-c(psi_choice, logQ_cstar))
ss_part = fit_resid(og_part, D, n_samps, prior)
ts_part = fit_resid(ss_part, D, n_samps / 2, prior)

log_sum_exp(unlist(compute_expterms(ss_part, D)))
log_sum_exp(unlist(compute_expterms(ts_part, D)))

hme = hme_approx(u_df, prior, J, D, N)

hml_approx$const_vec       # -272.1245
hme
came_approx(u_df, hml_approx, prior, post, J, D)
(LIL = lil(prior, post))   # -272.1202

mean(c(log_sum_exp(unlist(compute_expterms(ss_part, D))),
     log_sum_exp(unlist(compute_expterms(ts_part, D))),
     hml_approx$const_vec))


#### begin simulations ---------------------------------------------------------

B = 1000 # number of replications
J = 5000 # number of MCMC samples per replication

hyb_fs  = numeric(B) # store harmonic mean estiator
hyb_ss  = numeric(B) # store harmonic mean estiator
hyb_ts  = numeric(B) # store harmonic mean estiator
hyb     = numeric(B) # store harmonic mean estiator
hme     = numeric(B) # store hybrid estimator
ame     = numeric(B) # store arithmetic mean estimator
came    = numeric(B) # corrected arithmetic mean estimator

# other estimators: chib's, bridge, more recent ones?

for (b in 1:B) {
    
    # sample from posterior
    u_samps = rmvnorm(J, mean = c(mu_beta), sigma = Q_beta_inv) %>% data.frame 
    u_df = preprocess(u_samps, D, prior) # J x (D + 1) -- stored row-wise 
    
    #### (1) hybrid estimator
    hml_approx = hml_const(1, D, u_df, J, prior)
    og_part = hml_approx$param_out %>% 
        dplyr::select(-c(psi_choice, logQ_cstar))
    ss_part = fit_resid(og_part, D, n_samps, prior)
    ts_part = fit_resid(ss_part, D, n_samps / 2, prior)
    
    hyb_fs[b] = hml_approx$const_vec
    hyb_ss[b] = log_sum_exp(unlist(compute_expterms(ss_part, D)))
    hyb_ts[b] = log_sum_exp(unlist(compute_expterms(ts_part, D)))
    
    hyb[b] = mean(c(hyb_fs[b], hyb_ss[b], hyb_ts[b]))
    
    #### (2) harmonic mean estimator
    hme[b] = hme_approx(u_df, prior, J, D, N)
    
    #### (3) corrected arithmetic mean estimator (IS)
    came[b] = came_approx(u_df, hml_approx, prior, post, J, D)
    
    if (b %% 100 == 0) { print(paste("iter:", b)) }
}

approx_df = data.frame(mcmc = 1:B, hme = hme, came = came, hyb = hyb)
approx_long = melt(approx_df, id.vars = 'mcmc')

ggplot(approx_long, aes(x = mcmc, y = value, col = variable)) + geom_point() +
    geom_hline(aes(yintercept = LIL), linetype = 'dashed', size = 0.9)

### mean, error
(LIL = lil(prior, post))

round((mean(hyb) + median(hyb))/2, 3)
round(mean(hme), 3)
# mean(ame)
round(mean(came), 3)

round(mean(hyb_fs), 3)
round(mean(hyb_ss), 3)
round(mean(hyb_ts), 3)

mean(hyb)

round(mean(LIL - hyb), 3)
round(mean(LIL - hme), 3)
# mean(LIL - ame)
round(mean(LIL - came), 3)

round(sqrt(mean((LIL - hyb)^2)), 3)
round(sqrt(mean((LIL - hme)^2)), 3)
# sqrt(mean((LIL - ame)^2))
round(sqrt(mean((LIL - came)^2)), 3)












