

### create parameters for a single G_5 -----------------------------------------
set.seed(1)
p1 = 5
G_5 = matrix(c(1,1,0,1,1,
               1,1,1,0,0,
               0,1,1,1,1,
               1,0,1,1,1,
               1,0,1,1,1), p1, p1)

b = 5
delta = 0.5*(b-2)
n = 1
V = n * diag(1, p1)
V = BDgraph::rgwish(1, G, b, diag(p))
V = cov2cor(V)

I_G = function(delta) {
  7/2 * log(pi) + lgamma(delta + 5/2) - lgamma(delta + 3) +
    lgamma(delta + 1) + lgamma(delta + 3/2) + 2 * lgamma(delta + 2) +
    lgamma(delta + 5/2)
}
(Z_5  = log(2^(0.5*p1*b + 7)) + I_G(0.5*(b-2)) + (-0.5 * p1 * b - 7) * log(n))
Z = Z_5
Z
BDgraph::gnorm(G, b, V, J) # gnorm estimate of the entire (appended graph)

###### uhler formula implementation for general D for G_5

## rising factorial function
rf = function(a, k) {
  gamma(a + k) / gamma(a)
} # end rf() function

# final term in the innermost product of Eq. (4.1) before evaluation of
# derivative
f = function(t1, t2, q) {
  det_numer = det(V[c(4,5), c(4,5)] -
                    t(V[c(1,3), c(4,5)]) %*% solve(V[c(1,3), c(1,3)]) %*%
                    V[c(1,3), c(4,5)])^(delta + 3/2)
  tmp = V[c(4,5), c(4,5)] - (t1 * V[1, c(4,5)] %*% t(V[1, c(4,5)]) +
                               t2 * V[2, c(4,5)] %*% t(V[2, c(4,5)]))

  det_term = tmp[1,1] * tmp[2,2] - tmp[1,2] * tmp[2,1]

  det_numer / det_term^(delta + q + 3/2)
} # end f()

V_45 = V[c(4,5), c(4,5)]
V_1_45 = V[1, c(4,5)]
V_2_45 = V[2, c(4,5)]
h = function(t1, t2, q) {
  # det_numer = det(V[c(4,5), c(4,5)] -
  #                   t(V[c(1,3), c(4,5)]) %*% solve(V[c(1,3), c(1,3)]) %*%
  #                   V[c(1,3), c(4,5)])^(delta + 3/2)

  tmp = V_45 - (t1 * V_1_45 %*% t(V_1_45) + t2 * V_2_45 %*% t(V_2_45))
  det_term = tmp[1,1] * tmp[2,2] - tmp[1,2] * tmp[2,1]

  1 / det_term^(delta + q + 3/2)
} # end f()

d6h = Deriv(h, x = c("t1", "t2"), nderiv = 6)


## symbolic differentiation of f()
d1f = Deriv(f, x = c("t1", "t2"), nderiv = 1)
d2f = Deriv(f, x = c("t1", "t2"), nderiv = 2)
d3f = Deriv(f, x = c("t1", "t2"), nderiv = 3)
d4f = Deriv(f, x = c("t1", "t2"), nderiv = 4)
d5f = Deriv(f, x = c("t1", "t2"), nderiv = 5)
d6f = Deriv(f, x = c("t1", "t2"), nderiv = 6)


## (t1+t2) derivative of f() evaluated at (0,0)
d1f_00 = d1f(0, 0, q = 1)
d2f_00 = d2f(0, 0, q = 1)
d3f_00 = d3f(0, 0, q = 1)
d4f_00 = d4f(0, 0, q = 1)
d5f_00 = d5f(0, 0, q = 1)
d6h_00 = d6h(0, 0, q = 1)

IG5_I5 = I_G(0.5*(b-2))

## compute I_{G_5} (delta, V)
g = function(l_max, q_max, V) {

  q_sum = 0

  ## compute some constant
  V_13 = V[c(1,3), c(1,3)]
  c1 = (V_13[1,1] * V_13[2,2] - V_13[1,2] * V_13[2,1])^(2 * delta + 9/2) /
    ( det(V[c(1:3), c(1:3)])^(delta + 2) *
        det(V[c(1,3:5), c(1,3:5)])^(delta + 5/2) )

  V_1345 = V[c(1,3), c(4,5)]
  det_V_1345 = V_1345[1,1] * V_1345[2,2] - V_1345[1,2] * V_1345[2,1]

  V_45 = V[c(4,5), c(4,5)]
  tmp_numer = V_45 - t(V_1345) %*% solve(V_13) %*% V_1345
  det_numer = tmp_numer[1,1] * tmp_numer[2,2] - tmp_numer[1,2] * tmp_numer[2,2]

  for (q in 0:q_max) {

    # print(paste("q = ", q, sep = ""))

    # 1st term in summation over q (line 2 of Eq. 4.1)
    q_tmp = rf(delta + 5/2, q) * rf(delta + 3/2, q) /
      (factorial(q) * rf(delta + 3, 2*q)) * det_V_1345^(2*q)
    # print(q_tmp)

    ## update the functions to reflect a change in 'q'
    d1f_00 = d1f(0, 0, q = q)
    d2f_00 = d2f(0, 0, q = q)
    d3f_00 = d3f(0, 0, q = q)
    d4f_00 = d4f(0, 0, q = q)
    d5f_00 = d5f(0, 0, q = q)
    d6h_00 = d6h(0, 0, q = q)
    d7s_00 = d7s(0, 0, q = q)
    d8s_00 = d8s(0, 0, q = q)

    ## compute outer summation over l
    l_sum = 0
    for (l in 0:l_max) {
      l_tmp = 1 / (factorial(l) * rf(delta + 2*q + 3, l))
      # print(paste("    ", "l_tmp = ", l_tmp, sep = ""))
      ## compute inner summation over l1, l2
      l1 = 0
      l2 = l
      l12_sum = 0
      deriv_l1_l2 = 0

      while (l1 <= l) {

        # compute (l1, l2)-order mixed partial
        if (l == 0) {
          ### CASE 1: l = 0
          det_denom = V_45[1,1] * V_45[2,2] - V_45[1,2] * V_45[2,1]
          deriv_l1_l2 = det_numer^(delta + 3/2) / det_denom^(delta + q + 3/2)
        }
        else if (l == 1) {
          ### CASE 1: l = 0
          if (l1 == 1) {
            deriv_l1_l2 = d1f_00[1]
          } else if (l2 == 1) {
            deriv_l1_l2 = d1f_00[2]
          } else {
            print("this should not be reached")
          }
        }
        else if (l == 2) {
          ## CASE 2: l == 2
          if (l1 == 2) {
            deriv_l1_l2 = unname(d2f_00[1])
          } else if (l2 == 2) {
            deriv_l1_l2 = unname(d2f_00[4])
          } else if (l1 == 1 || l2 == 1) {
            deriv_l1_l2 = unname(d2f_00[2])
          } else { # l1 = 0, l2 = 0, i.e., evaluate w/o derivatives at (0,0)
            print("this should not be reached")
          }
        }
        else if (l == 3) {
          ## CASE 3: l == 3
          if (l1 == 3) {                        # l1 = 3, l2 = 0
            deriv_l1_l2 = unname(d3f_00[1])
          } else if (l1 == 2) {                 # l1 = 2, l2 = 1
            deriv_l1_l2 = unname(d3f_00[2])
          } else if (l1 == 1) {                 # l1 = 1, l2 = 2
            deriv_l1_l2 = unname(d3f_00[4])
          } else if (l1 == 0) {                 # l1 = 0, l2 = 3
            deriv_l1_l2 = unname(d3f_00[8])
          } else {
            print("this should be not be reached")
          }
        }
        else if (l == 4) {
          ## CASE 4: l == 4
          if (l1 == 4) {
            deriv_l1_l2 = d4f_00[1]
          } else if (l1 == 3) {
            deriv_l1_l2 = d4f_00[2]
          } else if (l1 == 2) {
            deriv_l1_l2 = d4f_00[4]
          } else if (l1 == 1) {
            deriv_l1_l2 = d4f_00[15]
          } else if (l1 == 0) {
            deriv_l1_l2 = d4f_00[16]
          } else {
            print("this should be not be reached")
          }
        }
        else if (l == 5) {
          if (l1 == 5) {
            deriv_l1_l2 = d5f_00[1]
          } else if (l1 == 4) {
            deriv_l1_l2 = d5f_00[2]
          } else if (l1 == 3) {
            deriv_l1_l2 = d5f_00[4]
          } else if (l1 == 2) {
            deriv_l1_l2 = d5f_00[29]
          } else if (l1 == 1) {
            deriv_l1_l2 = d5f_00[31]
          } else if (l1 == 0) {
            deriv_l1_l2 = d5f_00[32]
          } else {
            print("this should be not be reached")
          }
        }
        else if (l == 6) {
          tmp_numer = det_numer^(delta + 3/2)
          if (l1 == 6) {
            deriv_l1_l2 = tmp_numer * d6h_00[1]
          } else if (l1 == 5) {
            deriv_l1_l2 = tmp_numer * d6h_00[2]
          } else if (l1 == 4) {
            deriv_l1_l2 = tmp_numer * d6h_00[4]
          } else if (l1 == 3) {
            deriv_l1_l2 = tmp_numer * d6h_00[8]
          } else if (l1 == 2) {
            deriv_l1_l2 = tmp_numer * d6h_00[61]
          } else if (l1 == 1) {
            deriv_l1_l2 = tmp_numer * d6h_00[63]
          } else if (l1 == 0) {
            deriv_l1_l2 = tmp_numer * d6h_00[64]
          } else {
            print("should not be here")
          }
        }
        else if (l == 7) {

          tmp_numer = det_numer^(delta + 3/2)
          if (l1 == 7) {
            deriv_l1_l2 = tmp_numer * d7s_00[1]
          } else if (l1 == 6) {
            deriv_l1_l2 = tmp_numer * d7s_00[2]
          } else if (l1 == 5) {
            deriv_l1_l2 = tmp_numer * d7s_00[4]
          } else if (l1 == 4) {
            deriv_l1_l2 = tmp_numer * d7s_00[8]
          } else if (l1 == 3) {
            deriv_l1_l2 = tmp_numer * d7s_00[16]
          } else if (l1 == 2) {
            deriv_l1_l2 = tmp_numer * d7s_00[125]
          } else if (l1 == 1) {
            deriv_l1_l2 = tmp_numer * d7s_00[127]
          } else if (l1 == 0) {
            deriv_l1_l2 = tmp_numer * d7s_00[128]
          } else {
            print("should not be here")
          }

        }
        else if (l == 8) {
          tmp_numer = det_numer^(delta + 3/2)
          if (l1 == 8) {
            deriv_l1_l2 = tmp_numer * d8s_00[2^0]
          } else if (l1 == 7) {
            deriv_l1_l2 = tmp_numer * d8s_00[2^1]
          } else if (l1 == 6) {
            deriv_l1_l2 = tmp_numer * d8s_00[2^2]
          } else if (l1 == 5) {
            deriv_l1_l2 = tmp_numer * d8s_00[2^3]
          } else if (l1 == 4) {
            deriv_l1_l2 = tmp_numer * d8s_00[2^4]
          } else if (l1 == 3) {
            deriv_l1_l2 = tmp_numer * d8s_00[2^5]
          } else if (l1 == 2) {
            deriv_l1_l2 = tmp_numer * d8s_00[2^6]
          } else if (l1 == 1) {
            deriv_l1_l2 = tmp_numer * d8s_00[2^7]
          } else if (l1 == 0) {
            deriv_l1_l2 = tmp_numer * d8s_00[2^8]
          } else {
            print("should not be here")
          }

        }
        else {
          ## CASE 4: l > 8
          print("higher order not supported yet.")
        }

        l12_sum = l12_sum + choose(l, l1) *
          rf(delta + q + 5/2, l1) * rf(delta + q + 5/2, l2) *
          deriv_l1_l2

        ## increment/decrement indices
        l1 = l1 + 1
        l2 = l2 - 1
      } ## end inner summation over l1, l2

      l_sum = l_sum + l_tmp * l12_sum
      # print(paste("    ", "l = ", l, ": ", l12_sum, sep = ""))
    } # end 2nd summation over l

    # print(paste("    ", "q_contribution = ", q_tmp * l_sum, sep = ""))
    q_sum = q_sum + q_tmp * l_sum

  } # end 1st (outer) summation over q

  return(unname(IG5_I5 + log(c1) + log(q_sum)))

} # end g() --------------------------------------------------------------------

ll = 3
qq = 10
g(2, qq, V)
g(3, qq, V)
g(4, qq, V)
g(5, qq, V)
g(6, qq, V)
g(7, qq, V)
BDgraph::gnorm(G, b, V, J) - log(2^(0.5*p1*b + 7))

avg = 0
for (i in 1:100) {
  avg = avg + BDgraph::gnorm(G, b, V, J) - log(2^(0.5*p1*b + 7))
}
avg/100 # 11.23536

IG5_I5

V_45 = V[c(4,5), c(4,5)]
V_1345 = V[c(1,3), c(4,5)]
tmp_numer = V_45 - t(V_1345) %*% solve(V_13) %*% V_1345
det_numer = tmp_numer[1,1] * tmp_numer[2,2] - tmp_numer[1,2] * tmp_numer[2,2]

det_numer^(delta + 3/2) * deriv_denom


X = V[c(4,5), c(4,5)]
Y = V[1, c(4,5)]
Z = V[2, c(4,5)]


s = function(t1, t2, q) {
  ((X[1,1] - t1 * Y[1] * Y[1] - t2 * Z[1] * Z[1]) *
    (X[2,2] - t1 * Y[2] * Y[2] - t2 * Z[2] * Z[2]) -
    (X[1,2] - t1 * Y[1] * Y[2] - t2 * Z[1] * Z[2])^2)^(-delta - q - 3/2)
}

s = function(t1, t2, q) {
  tmp = X[1,1] * X[2,2] - t1 * X[1,1] * Y[2]^2 - t2 * X[1,1] * Z[2]^2 -
    t1 * Y[1]^2 * X[2,2] + t1^2 * Y[1]^2 * Y[2]^2 + t1 * t2 * Y[1]^2 * Z[2]^2 -
    t2 * Z[1]^2 * X[2,2] + t1 * t2 * Z[1]^2 * Y[2]^2 + t2^2 * Z[1]^2 * Z[2]^2 -
    X[1,2]^2 + 2 * t1 * X[1,2] * Y[1] * Y[2] + 2 * t2 * X[1,2] * Z[1] * Z[2] -
    2 * t1 * t2 * Y[1] * Y[2] * Z[1] * Z[2] - t1^2 * Y[1]^2 * Y[2]^2 -
    t2^2 * Z[1]^2 * Z[2]^2
    tmp^(-delta-q-3/2)
}

d1s = Deriv(s, x = c("t1", "t2"), nderiv = 1)
d2s = Deriv(s, x = c("t1", "t2"), nderiv = 2)
d3s = Deriv(s, x = c("t1", "t2"), nderiv = 3)

d7s = Deriv(s, x = c("t1", "t2"), nderiv = 7)
d8s = Deriv(s, x = c("t1", "t2"), nderiv = 8)
d9s = Deriv(s, x = c("t1", "t2"), nderiv = 9)


d1s_00 = d1s(0, 0, q = 1)
d2s_00 = d2s(0, 0, q = 1)
d3s_00 = d3s(0, 0, q = 1)

d7s_00 = d7s(0, 0, q = 1)

d1f_00
det_numer^(delta + 3/2) * d1s_00

d2f_00
det_numer^(delta + 3/2) * d2s_00

d3f_00
det_numer^(delta + 3/2) * d3s_00


g(2, V)
BDgraph::gnorm(G, b, V, J) - log(2^(0.5*p1*b + 7))








