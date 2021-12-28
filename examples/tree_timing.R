

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

## generate the data:
GG = graphml::init_graph(G, b, V)
samps = graphml::rgw(J, GG)
u_samps = samps %>% data.frame
# u_samps %>% head
psi = graphml::psi_cpp
u_df = hybridml::preprocess(u_samps, GG$D, GG) # J x (D_u + 1)

z = as.matrix(unname(cbind(u_df[,GG$D+1], u_df[,-(GG$D+1)])))

## use rpart to fit cart

r_func = function(u_df) {
  rpart(psi_u ~ ., u_df)
}

library(rpart.plot)
dim(rpart.rules(r_func(u_df)))
timeTree(z)


microbenchmark::microbenchmark(r     = r_func(u_df),
                               cpp   = timeTree(z),
                               times = 10
)



## use C++ implementation to fit cart






