setwd("C:/Users/ericc/Documents/graphml")
source("examples/helpers.R")
library(graphml)
library(rpart)
library(dplyr)

set.seed(1)
p = 5
G = matrix(c(1,1,0,1,1,
             1,1,1,0,0,
             0,1,1,1,1,
             1,0,1,1,1,
             1,0,1,1,1), p, p)
b = 300
V = BDgraph::rgwish(1, G, b, diag(p))
J = 1000
##### new implementation
set.seed(1)

BDgraph::gnorm(G, b, V, J)
graphml::gnorm_c(G, b, V, J)
graphml::gnormJT(G, getEdgeMat(G), b, V, J)
graphml::generalApprox(G, b, V, J)




