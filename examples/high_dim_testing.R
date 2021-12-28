

### 12/27: higher-dimensional testing --
### gnormJT() makes call to the hybrid algorithm on the
### sub-graphs

set.seed(1234)
p = 60
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
gnorm_c(Adj, b, D, 1000)
graphml::gnormJT(Adj, EdgeMat, b, D, 1000)





