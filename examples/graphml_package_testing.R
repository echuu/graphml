
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













