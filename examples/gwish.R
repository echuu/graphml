

# source("C:/Users/ericc/Documents/hybridml/examples/gwish/gwish_density.R")
# library(BDgraph)
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



# GG = initGeneralGraph(G, b, V)
# P = chol(solve(V))

# library(BDgraph)

##### new implementation
GG = graphml::init_graph(G, b, V)
J = 2000

samps = graphml::rgw(J, GG)
samps_psi = graphml::evalPsi(samps, GG)
u_df_cpp = data.frame(samps_psi)
# u_df_cpp %>% head
u_df_names = character(GG$D + 1)
for (d in 1:GG$D) {
  u_df_names[d] = paste("u", d, sep = '')
}
u_df_names[GG$D + 1] = "psi_u"
names(u_df_cpp) = u_df_names
u_star_cpp = graphml::calcMode(as.matrix(u_df_cpp), GG)
h(u_df_cpp, GG, GG$D, u_0 = u_star_cpp)
BDgraph::gnorm(G, b, V, 1000)


##### old implementation
GG = graphml::init_graph(G, b, V)
samps = rGW(J, GG)
u_samps = samps$Psi_free %>% data.frame
# u_samps %>% head
psi = graphml::psi_cpp
u_df = hybridml::preprocess(u_samps, GG$D, GG) # J x (D_u + 1)
u_star = graphml::calcMode(as.matrix(u_df), GG)
h(u_df, GG, GG$D, u_0 = u_star)

BDgraph::gnorm(G, b, V, 1000)

u = unname(unlist(u_df[1, 1:GG$D]))

## -----------------------------------------------------------------------------

## testing the code that comes after constructing the regression tree

u_rpart = rpart::rpart(psi_u ~ ., u_df_cpp)
param_support = graphml::support(samps, GG$D)

## this part still needs to be implemented in C++
u_partition = hybridml::extractPartition(u_rpart, param_support)

bounds = u_partition %>% dplyr::arrange(leaf_id) %>%
  dplyr::select(-c("psi_hat", "leaf_id"))

## ---------------------------------------------------


# candidates = findAllCandidatePoints(samps, unname(u_rpart$where),
#                                     u_star_cpp, GG$D)
#
# boundMap = createPartitionMap(t(bounds), u_partition$leaf_id)

cpp = function() {
  u_rpart = rpart::rpart(psi_u ~ ., u_df_cpp)
  param_support = graphml::support(samps, GG$D)

  ## this part still needs to be implemented in C++
  u_partition = hybridml::extractPartition(u_rpart, param_support)

  bounds = u_partition %>% dplyr::arrange(leaf_id) %>%
    dplyr::select(-c("psi_hat", "leaf_id"))

  approxWrapper(samps, unname(u_rpart$where), u_star_cpp, GG$D,
                t(bounds), u_partition$leaf_id, GG)
}

approxWrapper(samps, unname(u_rpart$where), u_star_cpp, GG$D,
              t(bounds), u_partition$leaf_id, GG)



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


u_df_part = u_df_cpp %>% dplyr::mutate(leaf_id = u_rpart$where)

l1_cost = apply(u_df_part[,1:GG$D], 1, hybridml::l1_norm, u_0 = u_star)
u_df_part = u_df_part %>% dplyr::mutate(l1_cost = l1_cost)

# take min result, group_by() leaf_id
psi_df = u_df_part %>%
  dplyr::group_by(leaf_id) %>% dplyr::filter(l1_cost == min(l1_cost)) %>%
  data.frame

r1 = function() {
  bounds = u_partition %>% dplyr::arrange(leaf_id) %>%
    dplyr::select(-c("psi_hat", "leaf_id"))
}

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
  r = r(),
  cpp = findAllCandidatePoints(samps, unname(u_rpart$where),
                               u_star_cpp, GG$D),
  times = 10
)

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




