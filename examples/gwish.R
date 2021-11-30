

source("C:/Users/ericc/Documents/hybridml/examples/gwish/gwish_density.R")
library(BDgraph)
library(dplyr)


set.seed(1)
p = 5
G = matrix(c(1,1,0,1,1,
             1,1,1,0,0,
             0,1,1,1,1,
             1,0,1,1,1,
             1,0,1,1,1), p, p)
b = 300
V = rgwish(1, G, b, diag(p))



GG = initGeneralGraph(G, b, V)

P = chol(solve(V))

library(BDgraph)
J = 2000

# Rcpp::sourceCpp("C:/Users/ericc/Documents/graphml/src/rgwishart.cpp")
GG = graphml::init_graph(G, b, V)
GG$FREE_PARAMS_ALL
J = 2000
samps = rGW(J, GG)

u_samps = samps$Psi_free %>% data.frame
u_df = hybridml::preprocess(u_samps, GG$D, GG) # J x (D_u + 1)
u_df %>% head


Rcpp::sourceCpp("C:/Users/ericc/Documents/hybridml/examples/gwish/gwish.cpp")

samps$Psi_free %>% dim
samps_mat = samps$Psi_free
graphml::evalPsi(samps_mat, GG) %>% head
u_df %>% head

BDgraph::gnorm(G, b, V, 100)

graphml::approxZ(G, b, V)


test = graphml::rgw(J, GG)
test %>% head

graphml::create_psi_mat_cpp(u, GG)
graphml::psi_cpp_mat(graphml::create_psi_mat_cpp(u, GG), GG)




microbenchmark::microbenchmark(r = hybridml::gwish_preprocess(u_samps, GG$D, GG),
                               cpp = evalPsi(samps_mat, GG),
                               times = 20)


calcMode(as.matrix(u_df), GG_cpp, VERBOSE = TRUE)
calcMode(as.matrix(u_df), GG_cpp)

u = unname(unlist(u_df[1, 1:GG$D]))


microbenchmark::microbenchmark(r = gwish_globalMode_mod(u_df, GG_cpp, GG_cpp,
                                                        psi = psi_cpp_mat,
                                                        grad = grad_gwish,
                                                        hess = hess_gwish),
                               cpp = calcMode(as.matrix(u_df), GG_cpp, VERBOSE = FALSE),
                               times = 20)


create_psi_mat_cpp(u, GG_cpp)

GG_cpp = init_graph(G, b, V)
x$t_ind
x$vbar
x$b_i
x$nu_i

GG$b_i
GG$nu_i

library(dplyr)

u_star = gwish_globalMode_mod(u_df, GG, GG,
                              psi = graphml::psi_cpp_mat,
                              grad = graphml::grad_gwish,
                              hess = graphml::hess_gwish)

u_star_cpp = graphml::calcMode(as.matrix(u_df), GG)


out = hybml_gwish_cpp(u_df, GG,
                      psi = graphml::psi_cpp_mat,
                      grad = graphml::grad_gwish,
                      hess = graphml::hess_gwish,
                      u_0 = u_star)

out$logz
gnorm(G, b, V, 100)

