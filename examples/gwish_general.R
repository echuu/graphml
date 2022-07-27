
## gwish_general.R
## author: Eric Chuu

## This file compares the performance of the GNORM and EPSOM-HybJT estimators.
## Just as we do in Section 4.3.2 in the paper, we randomly generate the graph
## with p vertices, where the existence of each edge is determined by a
## Bernoulli draw. Note, the scale matrix here is no assumed to be (block)
## diagonal, so we cannot use the formula for the normalizing constant from
## Uhler et al. (2016)

library(graphml) # load package for EPSOM-HybJT estimator
library(BDgraph) # load package for GNORM estimator
library(dplyr)   # load package for data manipulation

set.seed(1234)

p = 30 # number of vertices in the graph
# randomly generate the adjacency matrix for the graph
Adj = matrix(rbinom(p^2,1,0.15), p, p)
Adj = Adj + t(Adj)
diag(Adj) = 0
Adj[Adj==1]=0
Adj[Adj==2]=1
diag(Adj) = 1
EdgeMat = graphml::getEdgeMat(Adj)
b = 500 # degrees of freedom
Y = matrix(rnorm(p * 500), nrow = 500, ncol = p)
D = t(Y) %*% Y # scale matrix

# EPSOM-HybJT estimate of GW(b, D) log normalizing constant
graphml::hybridJT(Adj, EdgeMat, b, D, 1000)

# GNORM estimate of GW(b, V)  log normalizing constant
BDgraph::gnorm(Adj, b, D, 1000)
