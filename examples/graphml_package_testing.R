
graphml::calcMode(as.matrix(u_df_cpp), GG)

u = unname(unlist(u_df_cpp[1, 1:GG$D]))
u_mat = create_psi_mat_cpp(u, GG)

test_grad = graphml::grad_cpp_mat(u_mat, GG)
test_faster = graphml::grad_cpp_mat(u_mat, GG)
grad_gwish_test = graphml::grad_gwish(u_mat, GG)

all.equal(test_grad, grad_gwish_test)

## note: grad_gwish(), hess_gwish() are the GENERAL functions for non-diag
## cases --> do not use grad_cpp, hess_cpp functions anymore


microbenchmark::microbenchmark(
  r = h(u_df_cpp, GG, GG$D, u_0 = u_star_cpp),
  cpp = cpp(u_df_cpp, samps, samps_psi, GG, u_star_cpp),
  times = 20
)



cpp = function(u_df, samps, samps_psi, GG, u_star_cpp) {
  u_rpart = rpart::rpart(psi_u ~ ., u_df)
  param_support = graphml::support(samps, GG$D)

  ## this part still needs to be implemented in C++
  u_partition = hybridml::extractPartition(u_rpart, param_support)

  # bounds = u_partition %>%
  #   dplyr::select(-c("psi_hat", "leaf_id"))
  bounds = u_partition[,-c(1,2)]

  approxWrapper(samps_psi, unname(u_rpart$where), u_star_cpp, GG$D,
                t(bounds), u_partition$leaf_id, GG)
}



h = function(u_df, params, D, u_0 = NULL) {
  options(scipen = 999)
  options(dplyr.summarise.inform = FALSE)

  ## fit the regression tree via rpart()
  u_rpart = rpart::rpart(psi_u ~ ., u_df)

  ## (3) process the fitted tree
  # (3.1) obtain the (data-defined) support for each of the parameters
  param_support = hybridml::extractSupport(u_df, D) #

  # (3.2) obtain the partition
  u_partition = hybridml::extractPartition(u_rpart, param_support)

  #### hybrid extension begins here ------------------------------------------

  ### (1) find global mean
  # u_0 = colMeans(u_df[,1:D]) %>% unname() %>% unlist() # global mean

  if (is.null(u_0)) {
    MAP_LOC = which(u_df$psi_u == min(u_df$psi_u))
    u_0 = u_df[MAP_LOC,1:D] %>% unname() %>% unlist()
    # print(u_0)
  }

  ### (2) find point in each partition closest to global mean (for now)
  # u_k for each partition
  u_df_part = u_df %>% dplyr::mutate(leaf_id = u_rpart$where)

  l1_cost = apply(u_df_part[,1:D], 1, hybridml::l1_norm, u_0 = u_0)
  u_df_part = u_df_part %>% dplyr::mutate(l1_cost = l1_cost)

  # take min result, group_by() leaf_id
  psi_df = u_df_part %>%
    dplyr::group_by(leaf_id) %>% dplyr::filter(l1_cost == min(l1_cost)) %>%
    data.frame

  bounds = u_partition %>% dplyr::arrange(leaf_id) %>%
    dplyr::select(-c("psi_hat", "leaf_id"))
  psi_df = psi_df %>% dplyr::arrange(leaf_id)

  K = nrow(bounds)
  graphml::approx_integral(K, as.matrix(psi_df), as.matrix(bounds), params)
}










