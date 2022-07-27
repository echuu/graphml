

### 12/27: higher-dimensional testing --
### gnormJT() makes call to the hybrid algorithm on the
### sub-graphs

library(graphml)

set.seed(1234)
# p = 60
p = 30
Adj = matrix(rbinom(p^2,1,0.15), p, p)
Adj = Adj + t(Adj)
diag(Adj) = 0
Adj[Adj==1]=0
Adj[Adj==2]=1
diag(Adj) = 1
EdgeMat = getEdgeMat(Adj)
# JT = getJT(EdgeMat)
b = 500
Y = matrix(rnorm(p*500), nrow = 500, ncol = p)
D = t(Y)%*%Y

# BDgraph::gnorm(Adj, b, D, 1000)
# gnorm_c(Adj, b, D, 1000)

start_time <- Sys.time()
graphml::hybridJT(Adj, EdgeMat, b, D, 1000)
end_time <- Sys.time()
end_time - start_time

start_time <- Sys.time()
BDgraph::gnorm(Adj, b, D, 1000)
end_time <- Sys.time()
end_time - start_time


set.seed(1)
microbenchmark::microbenchmark(
  hybjt = graphml::hybridJT(Adj, EdgeMat, b, D, 1000),
  atay  = BDgraph::gnorm(Adj, b, D, 1000),
  times   = 20
)


n_sims = 20
hyb_approx = numeric(n_sims)
gnorm_approx = numeric(n_sims)
for (i in 1:n_sims) {
  hyb_approx[i] = graphml::hybridJT(Adj, EdgeMat, b, D, 1000)
  gnorm_approx[i] = BDgraph::gnorm(Adj, b, D, 1000)
}
mean(hyb_approx)
sd(hyb_approx)
mean(gnorm_approx)
sd(gnorm_approx)


# microbenchmark::microbenchmark(
#   hybjt = graphml::hybridJT(Adj, EdgeMat, b, D, 1000),
#   slow  = graphml::hybridJT_slow(Adj, EdgeMat, b, D, 1000),
#   times   = 20
# )





