
#include "Graph.h"


Graph::Graph(arma::umat G, u_int b, arma::mat V) {
    this->G = G;
    this->b = b;
    this->V = V;

    this->p = G.n_rows;
    this->P = chol(inv(V));
    this->P_inv = arma::inv(P);
    this->F = getFreeElem();
    this->free = vectorise(F);
    this->free_index = find(free);
            
    arma::uvec upInd = trimatu_ind(size(G));
    this->edgeInd = arma::conv_to<arma::vec>::from(G(upInd));

    this->k_i  = arma::conv_to<arma::vec>::from(arma::sum(F, 0) - 1);
	this->nu_i = arma::conv_to<arma::vec>::from(arma::sum(F, 1) - 1);
	this->b_i  = nu_i + k_i + 1;
	this->D    = arma::sum(edgeInd);

    // create the index matrix, t_ind: (TODO: move this to function)
    arma::uvec ind_vec = find(this->F > 0);
    arma::mat indexMatrix(D, 2, arma::fill::zeros);
    for (u_int d = 0; d < D; d++) {
        indexMatrix(d, 0) = ind_vec(d) % p; // row of free elmt
        indexMatrix(d, 1) = ind_vec(d) / p; // col of free elmt
	}
    // TODO: update the density functions so we can remove this
    indexMatrix = indexMatrix + 1;
    this->t_ind = indexMatrix;
    this->n_nonfree = p * (p + 1) / 2 - D;
    this->vbar = getNonFreeElem();
} // end of Graph constructor


arma::mat Graph::getFreeElem() {
    /*  set lower diag elmt of G to 0 so that we have the free elements
        on the diag + upper diag remaining; these are used to compute
        intermediate quantities such as k_i, nu_i, b_i; 
        see Atay paper for the 'A' matrix) */
    arma::mat F = arma::conv_to<arma::mat>::from(this->G);
    for (u_int r = 1; r < this->p; r++) {
        for (u_int c = 0; c < r; c++) {
            F(r, c) = 0;
        }
    }
    return F;
} // end getFreeElem() function


arma::mat Graph::getNonFreeElem() {
    /*  get the 2-column matrix that has row, column index of each of 
        the nonfree elements */
    arma::mat F = arma::conv_to<arma::mat>::from(this->G);
    for (u_int r = 0; r < (this->p - 1); r++) {
        for (u_int c = r + 1; c < this->p; c++) {
            if (F(r, c) == 0) {
                F(r, c) = -1;
            }
        }
    }
    arma::uvec ind_vbar = find(F < 0); // get ind of the nonfree elmts
    // TODO: think about when n_nonfree == 0 --> what does this mean? 
    // what happens to the subsequent calculations;
    arma::mat vbar(n_nonfree, 2, arma::fill::zeros);
    for (u_int n = 0; n < this->n_nonfree; n++) {
        vbar(n, 0) = ind_vbar(n) % this->p; // row of nonfree elmt
        vbar(n, 1) = ind_vbar(n) / this->p; // col of nonfree elmt
    }
    vbar = vbar + 1;
    return vbar;
} // end getFreeElem() function


// TODO: separate parallel / sequential functions -> put this into seq. file
arma::mat Graph::sampleGW(u_int m) {
    arma::mat G = arma::conv_to<arma::mat>::from(this->G);
    arma::mat samps(this->D, m, arma::fill::zeros);
    arma::mat omega, phi, zeta;
    arma::vec u0, u;
	arma::uvec ids  = this->free_index;
    for (unsigned int j = 0; j < m; j++) {
        omega = rgwish_c(G, this->P, this->b, this->p, 1e-8);
        phi   = arma::chol(omega);             // upper choleksy
        zeta  = phi * this->P_inv;             // compute transformation
        u0    = arma::vectorise(zeta);
        u     = u0(ids);                       // extract free elements
        samps.col(j) = u;
    } // end sampling loop
     return samps.t(); // return as a (m x D) matrix
} // end sampleGW() function

// TODO: separate parallel / sequential functions -> put this into parallel file
arma::mat Graph::sampleGWParallel(u_int J) {

	arma::mat G = arma::conv_to<arma::mat>::from(this->G);
    // arma::mat samps(graph->D, m, arma::fill::zeros);
	arma::mat samps(J, this->D, arma::fill::zeros);
	arma::mat P_inv = this->P_inv;

	std::vector<arma::vec> vec; 
	#pragma omp parallel shared(G)
    {
		arma::mat omega, phi, zeta;
		arma::vec u0, u;
		arma::uvec ids  = this->free_index;
		std::vector<arma::vec> vec_private; 
		#pragma omp for // fill vec_private in parallel
		for (unsigned int j = 0; j < J; j++) {
			omega = rgwish_c(G, this->P, this->b, this->p, 1e-8);
			phi   = arma::chol(omega);             // upper choleksy
			zeta  = phi * P_inv;             // compute transformation
			u0    = arma::vectorise(zeta);
			u     = u0(ids);                       // extract free elements
			// samps.col(j) = u;
			vec_private.push_back(u);
		} // end sampling loop
		#pragma omp critical
        vec.insert(vec.end(), vec_private.begin(), vec_private.end());
	} // end of outer omp

	for (unsigned int j = 0; j < J; j++) {
		arma::rowvec u = arma::conv_to< arma::rowvec >::from( vec[j] );
		samps.row(j) = u;
	}
     // return samps.t(); // return as a (m x D) matrix
	 return samps;
} // end sampleGW() function


