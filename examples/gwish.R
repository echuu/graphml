
# // [[Rcpp::plugins(openmp)]]

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



microbenchmark::microbenchmark(
  jt  = graphml::gnormJT(G, getEdgeMat(G), b, V, J),
  hyb = graphml::generalApprox(G, b, V, J),
  times = 20
)



regular = function() {
  set.seed(1)
  graphml::generalApprox(G, b, V, J)
}
fast = function() {
  set.seed(1)
  graphml::hybJT(G, b, V, J)
}

microbenchmark::microbenchmark(regular = regular(),
                               fast = fast(),
                               times = 10)





GG = graphml::init_graph(G, b, V)
set.seed(1)
J = 2000
samps = graphml::rgw(J, GG)
samps_psi = graphml::evalPsi(samps, GG)
u_df_cpp = graphml::mat2df(samps_psi, GG$df_name)
u_star_cpp = graphml::calcMode(samps_psi, GG)
approx_v1(u_df_cpp,
          u_star_cpp,
          samps_psi,
          GG)


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
# gnorm_c(Adj, b, D, 1000)
gnormJT(Adj, EdgeMat, b, D, 1000)



Adj_backup = Adj
Y_backup = Y
gnormJT(Adj_backup, EdgeMat, b, D, 1000)


# source("C:/Users/ericc/Documents/hybridml/examples/gwish/gwish_density.R")
# library(BDgraph)

## import R helper functions that will eventually be ported to C++
## contains h(), cpp(), and functions that extract the partition in matrix form
## from the rpart objects
#### THESE ALL NEED TO BE LOADED INTO THE GLOBAL ENVIRONMENT
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

##### new implementation
GG = graphml::init_graph(G, b, V)
set.seed(1)
J = 2000
samps = graphml::rgw(J, GG)
samps_psi = graphml::evalPsi(samps, GG)
u_df_cpp = graphml::mat2df(samps_psi, GG$df_name)
u_star_cpp = graphml::calcMode(samps_psi, GG)
approx_v1(u_df_cpp,
          u_star_cpp,
          samps_psi,
          GG)

set.seed(1)

BDgraph::gnorm(G, b, V, J)
graphml::gnorm_c(G, b, V, J)
graphml::gnormJT(G, getEdgeMat(G), b, V, J)

set.seed(1)
graphml::generalApprox(G, b, V, J)
set.seed(1)
graphml::hybJT(G, b, V, J)

old(G, b, V, J)



set.seed(2021)
p = 40;
G = matrix(0, p, p)
G[1, p] = 1
for(i in 1:(p-1)){
  G[i, i+1] = 1
}
G = G + t(G); diag(G) = 1
V = BDgraph::rgwish(1, G, b, diag(p))

graphml::gnormJT(G, getEdgeMat(G), b, V, J)

set.seed(1234)
p = 40
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

BDgraph::gnorm(Adj, b, D, 1000)
# gnorm_c(Adj, b, D, 1000)
gnormJT(Adj, EdgeMat, b, D, 1000)

# -10030.94




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
gnormJT(Adj, EdgeMat, b, D, 1000)


old(Adj, b, D, 1000)
microbenchmark::microbenchmark(# r = old(G, b, V, J),
                               cpp = gnorm_c(Adj, b, D, 1000),
                               jt = gnormJT(Adj, EdgeMat, b, D, 1000),
                               times = 10)








#### for donald: ---------------------------------------------------------------
## these are in approx_v1() function already; ideally these are replaced by
## pure C++ implementations rather than having to call back to R
tree = graphml::fitTree(u_df_cpp, psi_u ~.)
supp = support(samps, GG$D); ## already implemented
partList = extractPartitionSimple(tree, supp)

partList$leaf_id
partList$locs # location of each of the samples *in the order u_df_cpp*
partList$partition # partition sets *in the order of leaf_id*

#### for donald: ---------------------------------------------------------------





### TODO: rpart requires a dataframe, so we need column names for this --------
## fit cart
## get support
## get partition
## compute approximation using some wrapper function
approx_v1(u_df_cpp, psi_u~.,
          u_star_cpp,
          samps_psi,
          GG)

microbenchmark::microbenchmark(r = r(),
                               cpp = approx_v1(u_df_cpp, psi_u~.,
                                               u_star_cpp,
                                               samps_psi,
                                               GG),
                               times = 10)

test_mat = model.matrix(samps_psi)
samps_psi %>% head




library(rpart) # this must be loaded before calling the cpp function
library(dplyr)

old = function(G, b, V, J) {
  GG = graphml::init_graph(G, b, V)
  samps = graphml::rgw(J, GG)
  u_samps = samps%>% data.frame
  # u_samps %>% head
  psi = graphml::psi_cpp
  u_df = hybridml::preprocess(u_samps, GG$D, GG) # J x (D_u + 1)
  u_star = graphml::calcMode(as.matrix(u_df), GG)
  h(u_df, samps, GG, GG$D, u_0 = u_star)
}

r = function() {
  tree = graphml::fitTree(u_df_cpp, psi_u ~.)            ## calls to tools.cpp
  param_support = graphml::support(samps, GG$D)
  part = extractPartitionSimple(tree, param_support)

  locs = part$locs
  leaf_id = part$leaf_id
  bounds = part$partition
  approxWrapper(samps_psi, unname(tree$where), u_star_cpp, GG$D,
                bounds, leaf_id, GG)
}
r()






cpp(u_df_cpp, samps, samps_psi, GG, u_star_cpp)




extractPartition(tree, param_support)










h(u_df_cpp, samps, GG, GG$D, u_0 = u_star_cpp)
BDgraph::gnorm(G, b, V, 1000)


microbenchmark::microbenchmark(
  r = h(u_df_cpp, samps, GG, GG$D, u_0 = u_star_cpp),
  cpp = cpp(u_df_cpp, samps, samps_psi, GG, u_star_cpp),
  times = 10
)



###------
param_support = graphml::support(samps, GG$D)





test_const_arma(param_support)

source("examples/helpers.R")
library(dplyr)
extractPartition(tree, param_support)





bounds = tmpFunc(testpart)

u_rpart = rpart::rpart(psi_u ~ ., u_df_cpp)
param_support = graphml::support(samps, GG$D)

## this part still needs to be implemented in C++
u_partition = hybridml::extractPartition(u_rpart, param_support)

bounds = u_partition %>% dplyr::arrange(leaf_id) %>%
  dplyr::select(-c("psi_hat", "leaf_id"))







##### old implementation


BDgraph::gnorm(G, b, V, 1000)

u = unname(unlist(u_df[1, 1:GG$D]))

## -----------------------------------------------------------------------------

## testing the code that comes after constructing the regression tree


## ---------------------------------------------------


samps_psi %>% dim
candidates = findAllCandidatePoints(samps_psi, unname(u_rpart$where),
                                    u_star_cpp, GG$D)

boundMap = createPartitionMap(t(bounds), u_partition$leaf_id)

approxWrapper(samps_psi, unname(u_rpart$where), u_star_cpp, GG$D,
              t(bounds), u_partition$leaf_id, GG)


cpp = function() {
  u_rpart = rpart::rpart(psi_u ~ ., u_df_cpp)
  param_support = graphml::support(samps, GG$D)

  ## this part still needs to be implemented in C++
  u_partition = hybridml::extractPartition(u_rpart, param_support)

  bounds = u_partition %>% dplyr::arrange(leaf_id) %>%
    dplyr::select(-c("psi_hat", "leaf_id"))

  approxWrapper(samps_psi, unname(u_rpart$where), u_star_cpp, GG$D,
                t(bounds), u_partition$leaf_id, GG)
}



microbenchmark::microbenchmark(
  r = h(u_df_cpp, GG, GG$D, u_0 = u_star_cpp),
  cpp = cpp(),
  times = 10
)

########### work for 12/1 ------------------------------------------------------

u_df_part = u_df_cpp %>% dplyr::mutate(leaf_id = u_rpart$where)

l1_cost = apply(u_df_part[,1:GG$D], 1, hybridml::l1_norm, u_0 = u_star_cpp)
u_df_part = u_df_part %>% dplyr::mutate(l1_cost = l1_cost)

# take min result, group_by() leaf_id
psi_df = u_df_part %>%
  dplyr::group_by(leaf_id) %>% dplyr::filter(l1_cost == min(l1_cost)) %>%
  data.frame

## idea: pass in samps_psi results, where psi value is in the D-th column
## when finding index of min, also save the corresponding psi value into another
## hashmap that maps leaf -> psi_value





r = function() {
  u_df_part = u_df_cpp %>% dplyr::mutate(leaf_id = u_rpart$where)

  l1_cost = apply(u_df_part[,1:GG$D], 1, hybridml::l1_norm, u_0 = u_star_cpp)
  u_df_part = u_df_part %>% dplyr::mutate(l1_cost = l1_cost)
  # take min result, group_by() leaf_id
  psi_df = u_df_part %>%
    dplyr::group_by(leaf_id) %>% dplyr::filter(l1_cost == min(l1_cost)) %>%
    data.frame
  psi_df
}

microbenchmark::microbenchmark(
  r = r(),
  cpp = findAllCandidatePoints(samps_psi, unname(u_rpart$where),
                               u_star_cpp, GG$D),
  times = 10
)


r1 = function() {
  bounds = u_partition %>% dplyr::arrange(leaf_id) %>%
    dplyr::select(-c("psi_hat", "leaf_id"))
}

microbenchmark::microbenchmark(
  r = r1(),
  cpp = createPartitionMap(t(bounds), u_partition$leaf_id),
  times = 10
)


psi_df = psi_df %>% dplyr::arrange(leaf_id)

K = nrow(bounds)
graphml::approx_integral(K, as.matrix(psi_df), as.matrix(bounds), GG)


u_df_part$leaf_id %>% unique
cost = findCandidatePoint(test, u_star, GG$D)


test_func = findAllCandidatePoints(samps, unname(u_rpart$where),
                                   u_star, GG$D)
test_func = createPartitionMap(t(bounds), u_partition$leaf_id)



samps %>% head
test_func %>% dim




microbenchmark::microbenchmark(
  r = r1(),
  cpp = createPartitionMap(t(bounds), u_partition$leaf_id),
  times = 10
)




microbenchmark::microbenchmark(
  r = apply(u_df_part[,1:GG$D], 1, hybridml::l1_norm, u_0 = u_star),
  cpp = findCandidatePoint(test, u_star, GG$D),
  times = 30
)




