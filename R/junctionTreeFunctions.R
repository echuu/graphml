
# function to get the edge matrix that the junction tree part of the hybrid-ep
getEdgeMat = function(Adj){
  vAdj = apply(Adj, 2, sum)
  Adj[lower.tri(Adj, diag = FALSE)] = 0
  Edge = which(Adj==1, arr.ind = TRUE)

  l = c(rep(2, nrow(Edge)), rep(1, length(which(vAdj==0))))
  ind1 = c(Edge[,1], which(vAdj==0))
  ind2 = c(Edge[,2], rep(0,length(which(vAdj==0))))
  EdgeMat = cbind(l, ind1, ind2)
  colnames(EdgeMat) = NULL
  return(EdgeMat)
}


