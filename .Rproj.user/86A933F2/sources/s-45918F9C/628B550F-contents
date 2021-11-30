
## HIW_helper.R

## sampleHIW() function
#  J       : number of posterior samples to generate
#  D_u     : dim of the sample u that is fed into the tree (# nonzero in L')
#  D_0     : num entries on diagonal and upper diagonal (can remove later)
#  b       : df for HIW prior
#  N       : sample size
#  V       : (D x D) p.d. matrix for HIW Prior
#  S       : Y'Y, a (D x D) matrix
sampleHIW = function(J, D_u, D_0, G, b, N, V, S, edgeIndex) {

    Sigma_post    = vector("list", J) # store posterior samples in matrix form
    Omega_post    = vector("list", J) # store posterior samples in matrix form
    Lt_post       = vector("list", J) # store lower cholesky factor

    post_samps_0  = matrix(0, J, D_0) # store ENTIRE upper diag in vector form
    post_samps    = matrix(0, J, D_u) # store NONZERO upper diag in vector form

    for (j in 1:J) {

        # perform one draw from HIW(b + N, V + S)
        HIW_draw = HIWsim(G, b + N, V + S)
        draw_Sigma = HIW_draw$Sigma
        draw_Omega = HIW_draw$Omega

        Lt_post_j = chol(draw_Omega) # upper cholesky factor for Omega = LL'

        Sigma_post[[j]] = draw_Sigma                                # (D x D)
        Omega_post[[j]] = draw_Omega                                # (D x D)
        Lt_post[[j]]    = Lt_post_j                                 # (D x D)

        # collapse upper triangular matrix into a vector
        Lt_upper_vec = Lt_post_j[upper.tri(Lt_post_j, diag = T)]    # (D_0 x 1)

        # store entire vector (includes 0 elements)
        post_samps_0[j,]  = Lt_upper_vec                            # (D_0 x 1)

        # store nonzero vector (this is used to fit the tree)       # (D_u x 1)
        post_samps[j,] = Lt_upper_vec[edgeIndex]

    } # end of sampling loop


    ## note: Both post_samps_0 and post_samps can be used to re-construct
    ##       Lt_post. The former just needs to be stuck into the upper.tri
    ##       elements of a (D x D) matrix, while the latter requires the
    ##       edgeIndex logical vector to index into the correct elements

    return(list(post_samps   = data.frame(post_samps),
                post_samps_0 = data.frame(post_samps_0),
                Lt_post      = Lt_post,
                Sigma_post   = Sigma_post,
                Omega_post   = Omega_post, Lt_post = Lt_post))

} # end sampleHIW() function ---------------------------------------------------



## maxLogLik() function --------------------------------------------------------
# maximized log-likelihood, i.e., log-likelihood evaluated at the true Omega
maxLogLik = function(Omega, params) {

    N     = params$N      # number of observations
    D     = params$D      # num cols/rows in covariance matrix Sigma
    S     = params$S      # sum_n x_n x_n'

    loglik = - 0.5 * N * D * log(2 * pi) + 0.5 * N * log_det(Omega) -
        0.5 * matrixcalc::matrix.trace(Omega %*% S)

    return(loglik)

} # end maxLogLik() function ---------------------------------------------------





# ## HIW_loglik() function -----------------------------------------------------
# HIW_loglik_old = function(u, params) {
#
#     N = params$N
#     D = params$D
#     S = params$S
#
#     Lt = matrix(0, D, D)              # (D x D) lower triangular matrix
#     Lt[upper.tri(Lt, diag = T)] = u   # populate lower triangular terms
#
#     # recall: Sigma^(-1) = LL'
#
#     logprior = - 0.5 * N * D * log(2 * pi) + N * log_det(Lt) -
#         0.5 * matrix.trace(t(Lt) %*% Lt %*% S)
#
#     return(logprior)
#
# } # end HIW_loglik() function ------------------------------------------------





HIW_loglik = function(u, params) {

    N   = params$N
    D   = params$D
    D_0 = params$D_0
    S   = params$S

    Lt = matrix(0, D, D)     # (D x D) lower triangular matrix
    Lt_vec_0 = numeric(D_0)  # (D_0 x 1) vector to fill upper triangular, Lt
    Lt_vec_0[edgeInd] = u
    Lt[upper.tri(Lt, diag = T)] = Lt_vec_0   # populate lower triangular terms

    # recall: Sigma^(-1) = LL'

    logprior = - 0.5 * N * D * log(2 * pi) + N * log_det(Lt) -
        0.5 * matrixcalc::matrix.trace(t(Lt) %*% Lt %*% S)

    return(logprior)

} # end HIW_loglik() function --------------------------------------------------


## HIW_logprior() function -----------------------------------------------------
HIW_logprior = function(u, params) {

    # steps:
    # (1) extract upper diagonal entries
    # (2) extract diagonal entries
    # (3) compute nu = (nu_1,...,nu_p)
    # (4) compute upper diagonal part of log prior
    # (5) compute diagonal part of log prior

    D   = params$D              # number of rows/cols in Sigma/Omega
    D_0 = params$D_0            # num entries on diagonal and upper diagonal
    b   = params$b              # degrees of freedom

    edgeInd  = params$edgeInd   # indicator for present edges in the graph F
    upperInd = params$upperInd  # indicator for upper diagonal edges

    Lt = matrix(0, D, D)
    Lt_vec_0 = numeric(D_0)
    Lt_vec_0[edgeInd] = u
    Lt[upper.tri(Lt, diag = T)] = Lt_vec_0 # reconstruct Lt (upper tri matrix)

    #### (1) upper diagonal entries
    u_upper_all = Lt[upper.tri(Lt)] # extract upper diagonal entries
    u_upper = u_upper_all[upperInd] # keep only entries that have edge in G

    #### (2) diagonal entries
    u_diag = diag(Lt)               # extract diagonal entries, all included

    #### (3) compute nu_i, i = 1,..., d
    # compute nu_i (i = 1,...,D) by counting nonzero elements
    # in each row of Lt - 1
    # recall: the i-th row of Lt has exactly nu_i + 1 nonzero entries
    nu = rowSums(Lt != 0) - 1
    # nu[D] = 0

    #### (4) compute upper diagonal part of log prior
    upper_diag_prior = sum(- 0.5 * log(2 * pi) - 0.5 * u_upper^2)

    #### (5) compute diagonal part of log prior
    diag_prior = sum(- 0.5 * (b + nu) * log(2) - lgamma(0.5 * (b + nu)) +
                         (b + nu - 2) * log(u_diag) -
                         0.5 * u_diag^2 + log(2 * u_diag))

    # log prior, as shown in (8) of 3.2.2 HW Induced Cholesky Factor Density
    logprior = upper_diag_prior + diag_prior

    return(logprior)


} # end HIW_logprior() function ------------------------------------------------




## psi() function  -------------------------------------------------------------
old_psi = function(u, params) {

    loglik = HIW_loglik(u, params)
    logprior = HIW_logprior(u, params)


    return(- loglik - logprior)

} # end of psi() function ------------------------------------------------------



preprocess = function(post_samps, D, params) {

    psi_u = apply(post_samps, 1, old_psi, params = params) %>% unname() # (J x 1)

    # (1.2) name columns so that values can be extracted by partition.R
    u_df_names = character(D + 1)
    for (d in 1:D) {
        u_df_names[d] = paste("u", d, sep = '')
    }
    u_df_names[D + 1] = "psi_u"

    # populate u_df
    u_df = cbind(post_samps, psi_u) # J x (D + 1)
    names(u_df) = u_df_names


    return(u_df)

} # end of preprocess() function -----------------------------------------------





#####################################################################
############# HELPER FUNCTIONS FOR CALCULATING LOG MARG LIK #########
#####################################################################

# log multivariate gamma function Gamma_p(a)
logmultigamma = function(p, a){
    f = 0.25*p*(p-1)*log(pi)
    for(i in 1:p){ f = f + lgamma(a+0.5-0.5*i) }
    return(f)
}

logfrac = function(dim, b, D){
    temp = b+dim-1
    logfrac = 0.5*temp*log(det(D)) - 0.5*temp*dim*log(2) -
        logmultigamma(dim, 0.5*temp)
    return(logfrac)
}

# log normalizing constant for HIW
logHIWnorm = function(G, b, D){
    junct = makedecompgraph(G)
    cliques = junct$C; separators = junct$S
    nc = length(cliques); ns = length(separators)

    Cnorm = 0
    for(i in 1:nc){
        ind = cliques[[i]]
        Cnorm = Cnorm + logfrac(length(ind), b, D[ind, ind, drop = FALSE])
    }

    Snorm = 0
    if(ns>1){
        for(i in 2:ns){
            ind = separators[[i]]
            if (length(ind) != 0) {
                Snorm = Snorm + logfrac(length(ind), b, D[ind, ind, drop = FALSE])
            }
        }
    }

    logHIWnorm = Cnorm - Snorm
    return(logHIWnorm)
}

# # log marginal likelihood log(f(Y|G))
# logmarginal = function(Y, G, b, D){
#     n = nrow(Y); p = ncol(Y); S = t(Y)%*%Y
#     logmarginal = -0.5*n*p*log(2*pi) + logHIWnorm(G, b, D) -
#         logHIWnorm(G, b+n, D+S)
#     return(logmarginal)
# }

logmarginal = function(Y, G, b, D, S){
    n = nrow(Y); p = ncol(Y); # S = t(Y)%*%Y
    logmarginal = -0.5*n*p*log(2*pi) + logHIWnorm(G, b, D) -
        logHIWnorm(G, b+n, D+S)
    return(logmarginal)
}



# inverse Wishart density
logSingle = function(dim, b, D, Sigma){
    temp = b + dim - 1
    logSingle = 0.5 * temp * log(det(D)) - 0.5 * temp * dim * log(2) -
        logmultigamma(dim, 0.5 * temp)
    logSingle = logSingle - 0.5 * (b + 2 * dim) * log(det(Sigma)) -
        0.5 * sum(diag( solve(Sigma) %*% D ))

    return(logSingle)
}

# HIW density
logHIW = function(G, b, D, Sigma){
    junct = makedecompgraph(G)
    cliques = junct$C; separators = junct$S
    nc = length(cliques); ns = length(separators)

    Cnorm = 0
    for(i in 1:nc){
        ind = cliques[[i]]
        Cnorm = Cnorm + logSingle(length(ind), b,
                                  D[ind, ind, drop = FALSE],
                                  Sigma[ind, ind, drop = FALSE])
    }

    Snorm = 0
    if(ns>1){
        for(i in 2:ns){
            ind = separators[[i]]
            Snorm = Snorm + logSingle(length(ind), b,
                                      D[ind, ind, drop = FALSE],
                                      Sigma[ind, ind, drop = FALSE])
        }
    }

    logHIW = Cnorm - Snorm
    return(logHIW)
}





HIWsim = function(Adj, bG, DG){
    # check if "Adj" is a matrix object
    if(is.matrix(Adj)==FALSE) { stop("Adj must be a matrix object!") }
    # check if "Adj" is a square matrix
    if(dim(Adj)[1]!=dim(Adj)[2]) { stop("Adj must be a square matrix") }
    # check if "Adj" is symmetric
    if(isSymmetric.matrix(Adj)==FALSE) { stop("Adj must be a symmetric matrix") }

    # check if "DG" is a matrix object
    if(is.matrix(DG)==FALSE) { stop("DG must be a matrix object!") }
    # check if "DG" is a square matrix
    if(dim(DG)[1]!=dim(DG)[2]) { stop("DG must be a square matrix") }
    # check if "DG" is symmetric
    if(isSymmetric.matrix(DG)==FALSE) { stop("DG must be a symmetric matrix") }

    # check if "bG" is greater than 2
    if(bG<=2) { stop("bG must be greater than 2") }
    # check if "Adj" and "DG" are the same size
    if(nrow(Adj)!=nrow(DG)) { stop("Adj and DG must have the same dimension") }
    rMNorm = function(m, V){ return(m+t(chol(V))%*%rnorm(length(m))) }

    p = nrow(Adj)
    temp = makedecompgraph(Adj)
    Cliques = temp$C
    Separators = temp$S
    numberofcliques = length(Cliques)

    ############################################################
    # Creat some working arrays that are computed only once
    C1 = solve(DG[Cliques[[1]], Cliques[[1]]]/bG)
    c1 = Cliques[[1]]
    UN = c1
    DSi = DRS = mU = list()

    for(i in 2:numberofcliques){
        sid = Separators[[i]]
        if(length(sid)==0){
            DSi[[i]] = integer(0)
            cid = Cliques[[i]]
            dif = sort(setdiff(cid, UN))
            UN = sort(union(cid, UN)) # no need to sort, just playing safe
            sizedif = length(dif)
            DRS[[i]] = DG[dif, dif]
            DRS[[i]] = ( DRS[[i]] + t(DRS[[i]]) )/2
            mU[[i]] = integer(0)
        }
        else{
            DSi[[i]] = solve(DG[sid, sid])
            cid = Cliques[[i]]
            dif = sort(setdiff(cid, UN))
            UN = sort(union(cid, UN)) # no need to sort, just playing safe
            sizedif = length(dif)
            DRS[[i]] = DG[dif, dif] - DG[dif, sid] %*% DSi[[i]] %*% DG[sid, dif]
            DRS[[i]] = ( DRS[[i]] + t(DRS[[i]]) )/2
            mU[[i]] = DG[dif, sid] %*% DSi[[i]]
        }

    }

    ############################################################
    # MC Sampling
    UN = c1
    Sigmaj = matrix(0, p, p)
    # sample variance mx on first component
    Sigmaj[c1, c1] = solve(Wishart_InvA_RNG( bG+length(Cliques[[1]])-1, DG[Cliques[[1]], Cliques[[1]]] ))

    for(i in 2:numberofcliques){ # visit components and separators in turn
        dif = sort(setdiff(Cliques[[i]], UN))
        UN = sort(union(Cliques[[i]], UN)) # probably no need to sort, just playing safe
        sizedif = length(dif)
        sid = Separators[[i]]
        if(length(sid)==0){
            SigRS = solve(Wishart_InvA_RNG( bG+length(Cliques[[i]])-1, DRS[[i]] ))
            Sigmaj[dif, dif] = SigRS
        }
        else{
            SigRS = solve(Wishart_InvA_RNG( bG+length(Cliques[[i]])-1, DRS[[i]] ))
            Ui = rMNorm( as.vector(t(mU[[i]])), kronecker(SigRS, DSi[[i]]))
            Sigmaj[dif, sid] = t(matrix(Ui, ncol = sizedif)) %*% Sigmaj[sid, sid]
            Sigmaj[sid, dif] = t(Sigmaj[dif, sid])
            Sigmaj[dif, dif] = SigRS + Sigmaj[dif, sid] %*% solve(Sigmaj[sid, sid]) %*% Sigmaj[sid, dif]
        }
    }

    # Next, completion operation for sampled variance matrix
    H = c1
    for(i in 2:numberofcliques){
        dif = sort(setdiff(Cliques[[i]], H))
        sid = Separators[[i]]
        if(length(sid)==0){
            h = sort(setdiff(H, sid))
            Sigmaj[dif, h] = 0
            Sigmaj[h, dif] = t(Sigmaj[dif, h])
            H = sort(union(H, Cliques[[i]])) # probably no need to sort, just playing safe
        }
        else{
            h = sort(setdiff(H, sid))
            Sigmaj[dif, h] = Sigmaj[dif, sid] %*% solve(Sigmaj[sid, sid]) %*% Sigmaj[sid, h]
            Sigmaj[h, dif] = t(Sigmaj[dif, h])
            H = sort(union(H, Cliques[[i]])) # probably no need to sort, just playing safe
        }
    }
    Sigma = Sigmaj

    # Next, computing the corresponding sampled precision matrix
    Caux = Saux = array(0, c(p, p, numberofcliques))
    cid = Cliques[[1]]
    Caux[cid, cid, 1] = solve(Sigmaj[cid, cid])
    for(i in 2:numberofcliques){
        cid = Cliques[[i]]
        Caux[cid, cid, i] = solve(Sigmaj[cid, cid])
        sid = Separators[[i]]
        if(length(sid)!=0){
            Saux[sid, sid, i] = solve(Sigmaj[sid, sid])
        }
    }
    Omega = rowSums(Caux, dims = 2) - rowSums(Saux, dims = 2)

    return(list(Sigma = Sigma, Omega = Omega))
}




# Sample the HIW_G(bG,DG) distribution on a graph G with adjacency matrix Adj
# HIWsim = function(Adj, bG, DG){
#     # check if "Adj" is a matrix object
#     if(is.matrix(Adj)==FALSE) { stop("Adj must be a matrix object!") }
#     # check if "Adj" is a square matrix
#     if(dim(Adj)[1]!=dim(Adj)[2]) { stop("Adj must be a square matrix") }
#     # check if "Adj" is symmetric
#     if(isSymmetric.matrix(Adj)==FALSE) { stop("Adj must be a symmetric matrix") }
#
#     # check if "DG" is a matrix object
#     if(is.matrix(DG)==FALSE) { stop("DG must be a matrix object!") }
#     # check if "DG" is a square matrix
#     if(dim(DG)[1]!=dim(DG)[2]) { stop("DG must be a square matrix") }
#     # check if "DG" is symmetric
#     if(isSymmetric.matrix(DG)==FALSE) { stop("DG must be a symmetric matrix") }
#
#     # check if "bG" is greater than 2
#     if(bG<=2) { stop("bG must be greater than 2") }
#     # check if "Adj" and "DG" are the same size
#     if(nrow(Adj)!=nrow(DG)) { stop("Adj and DG must have the same dimension") }
#     rMNorm = function(m, V){ return(m+t(chol(V))%*%rnorm(length(m))) }
#
#     p = nrow(Adj)
#     temp = makedecompgraph(Adj)
#     Cliques = temp$C
#     Separators = temp$S
#     numberofcliques = length(Cliques)
#
#     ############################################################
#     # Creat some working arrays that are computed only once
#     C1 = solve(DG[Cliques[[1]], Cliques[[1]]]/bG)
#     c1 = Cliques[[1]]
#     UN = c1
#     DSi = DRS = mU = list()
#
#     for(i in 2:numberofcliques){
#         sid = Separators[[i]]
#         DSi[[i]] = solve(DG[sid, sid])
#         cid = Cliques[[i]]
#         dif = sort(setdiff(cid, UN))
#         UN = sort(union(cid, UN)) # no need to sort, just playing safe
#         sizedif = length(dif)
#         DRS[[i]] = DG[dif, dif] - DG[dif, sid] %*% DSi[[i]] %*% DG[sid, dif]
#         DRS[[i]] = ( DRS[[i]] + t(DRS[[i]]) )/2
#         mU[[i]] = DG[dif, sid] %*% DSi[[i]]
#     }
#
#     ############################################################
#     # MC Sampling
#     UN = c1
#     Sigmaj = matrix(0, p, p)
#     # sample variance mx on first component
#     Sigmaj[c1, c1] = solve(Wishart_InvA_RNG( bG+length(Cliques[[1]])-1, DG[Cliques[[1]], Cliques[[1]]] ))
#
#     for(i in 2:numberofcliques){ # visit components and separators in turn
#         dif = sort(setdiff(Cliques[[i]], UN))
#         UN = sort(union(Cliques[[i]], UN)) # probably no need to sort, just playing safe
#         sizedif = length(dif)
#         sid = Separators[[i]]
#         SigRS = solve(Wishart_InvA_RNG( bG+length(Cliques[[i]])-1, DRS[[i]] ))
#         Ui = rMNorm( as.vector(t(mU[[i]])), kronecker(SigRS, DSi[[i]]))
#         Sigmaj[dif, sid] = t(matrix(Ui, ncol = sizedif)) %*% Sigmaj[sid, sid]
#         Sigmaj[sid, dif] = t(Sigmaj[dif, sid])
#         Sigmaj[dif, dif] = SigRS + Sigmaj[dif, sid] %*% solve(Sigmaj[sid, sid]) %*% Sigmaj[sid, dif]
#     }
#
#     # Next, completion operation for sampled variance matrix
#     H = c1
#     for(i in 2:numberofcliques){
#         dif = sort(setdiff(Cliques[[i]], H))
#         sid = Separators[[i]]
#         h = sort(setdiff(H, sid))
#         Sigmaj[dif, h] = Sigmaj[dif, sid] %*% solve(Sigmaj[sid, sid]) %*% Sigmaj[sid, h]
#         Sigmaj[h, dif] = t(Sigmaj[dif, h])
#         H = sort(union(H, Cliques[[i]])) # probably no need to sort, just playing safe
#     }
#     Sigma = Sigmaj
#
#     # Next, computing the corresponding sampled precision matrix
#     Caux = Saux = array(0, c(p, p, numberofcliques))
#     cid = Cliques[[1]]
#     Caux[cid, cid, 1] = solve(Sigmaj[cid, cid])
#     for(i in 2:numberofcliques){
#         cid = Cliques[[i]]
#         Caux[cid, cid, i] = solve(Sigmaj[cid, cid])
#         sid = Separators[[i]]
#         Saux[sid, sid, i] = solve(Sigmaj[sid, sid])
#     }
#     Omega = rowSums(Caux, dims = 2) - rowSums(Saux, dims = 2)
#
#     return(list(Sigma = Sigma, Omega = Omega))
# }


# Input:  an adjacency  matrix A of a decomposable graph G
# Output: cell array G containing the cliques and separators of G
#         nodeIDs and nodenames are optional inputs

makedecompgraph = function(Adj){
    # first check if "Adj" is a matrix object
    if(is.matrix(Adj)==FALSE) { stop("the input must be a matrix object!") }
    # check if "Adj" is a square matrix
    if(dim(Adj)[1]!=dim(Adj)[2]) { stop("the input must be a square matrix") }
    # check if "Adj" is symmetric
    if(isSymmetric.matrix(Adj)==FALSE) { stop("the input must be a symmetric matrix") }

    p = nrow(Adj)
    Adj[Adj!=0] = 1 # set all non-zero entries of Adj to 1
    Adj = Adj - diag(diag(Adj)) + diag(p) # set all diagonal entries of Adj to be 1
    Order = 1:p
    i = 1
    Adj0 = Adj
    while(i<p){
        nn = apply(Adj[1:i, (i+1):p, drop=F], 2, sum)
        b = which.max(nn)
        Order[c(i+1, b+i)] = Order[c(b+i, i+1)]
        i = i + 1
        Adj = Adj0
        Adj = Adj[Order, Order]
    }

    numberofcliques = 1
    Cliques = list(1)
    i = 2
    while(i<=p){
        if( sum(Adj[i,Cliques[[numberofcliques]]])==length(Cliques[[numberofcliques]]) ){
            Cliques[[numberofcliques]] = c(Cliques[[numberofcliques]], i)
        }
        else{
            numberofcliques = numberofcliques + 1
            Cliques[[numberofcliques]] = union(i, which(Adj[i, 1:i]==1))
        }
        i = i + 1
    }

    for(i in 1:numberofcliques){
        Cliques[[i]] = sort(Order[Cliques[[i]]])
    }

    UN = Cliques[[1]]
    Separators = list()
    if(numberofcliques==1){ return(list(C = Cliques, S = Separators)) }
    else{
        for(i in 2:numberofcliques){
            Separators[[i]] = sort(intersect(UN, Cliques[[i]]))
            UN = union(UN, Cliques[[i]])
        }
        return(list(C = Cliques, S = Separators))
    }
}



# Generates 1 draw from a Wishart distribution - allows for singular Wishart too,
# in cases that the distn parameter S is rank deficient of rank r.

# K is W_p(df,A) with sum-of-squares parameter S = A^{-1} and d.o.f. df
# Dimension p is implicit

# Usual nonsingular case: r=p<df; now allow for r<p  with r integral
# Note that  E(K)=df.S^{-1}  in this notation

# Noticing pdf is p(K) = cons. |K|^((df-p-1)/2) exp(-trace(K S)/2)
# in case usual modification

# Returns matrix W of dimension p by p of rank r=rank(S)

# EXAMPLE: reference posterior in a normal model N(0,Sigma) with precision
# matrix Omega = Sigma^{-1}
# Random sample of size n has sample var matrix V=S/n with S=\sum_i x_ix_i'
# Ref posterior for precision mx Omega is W_p(n,A) with A=S^{-1}
# e.g., K = wishart_InvA_rnd(n,n*V); draw 1000 samples

# Useful for looking at posteriors for correlations and eigenvalues
# and also for posterior on graph - looking for elements of Omega near 0

# K = Wishart_InvA_RNG(df, S) <=> K ~ Wishart_p(df, S^{-1})
# E[K] = df * S^{-1}
Wishart_InvA_RNG = function(df, S){
    # first check if "S" is a matrix object
    if(is.matrix(S)==FALSE) { stop("the input must be a matrix object!") }
    # check if "S" is a square matrix
    if(dim(S)[1]!=dim(S)[2]) { stop("the input must be a square matrix") }
    # check if "S" is symmetric
    if(isSymmetric.matrix(S)==FALSE) { stop("the input must be a symmetric matrix") }

    p = nrow(S)
    temp = svd(S)
    P = temp$u
    D = diag(1/sqrt(temp$d), nrow = length(temp$d))
    i = which(temp$d>max(temp$d)*1e-9)
    r = length(i)
    P = P[,i] %*% D[,i]
    U = matrix(0, nrow = p, ncol = r)
    for(j in 1:r){
        U[j, j:r] = c(sqrt( rgamma(1, shape = (df-j+1)/2, scale = 2) ), rnorm(r-j))
    }
    U = U %*% t(P)
    K = t(U) %*% U
    return(K)
}






# end HIW_helper.R
