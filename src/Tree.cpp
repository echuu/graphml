
#include "Tree.h"

/* ----------------------- START TREE BUILDING ALGO #1 -----------------------*/
/* TREE BUILDING ALGO #1: sort the d-th feature for every node --> slowest */
Tree::Tree(arma::mat df) {

    this->z         = df;
    this->numRows   = df.n_rows;
    this->numFeats  = df.n_cols - 1;
    this->rowIds = arma::conv_to<arma::uvec>::from(
            arma::linspace(0, this->numRows-1, this->numRows));
    this->numLeaves = 0;
    this->numSplits = 0; 
    // this is the SSE for a tree with no splits
    this->treeSSE   = sse(df.col(0), numRows, arma::mean(df.col(0)));

    /* terminating condition variables */
    this->nodeMinLimit = 20; // min # obs in a node to consider a split
    this->minBucket    = nodeMinLimit / 3;
    this->cp           = 0.01;

    /* initialize variables related to extracting the partition */
    arma::uvec r = arma::conv_to<arma::uvec>::from(
            arma::linspace(0, this->numRows-1, this->numRows));
    arma::uvec c = arma::conv_to<arma::uvec>::from(
            arma::linspace(1, this->numFeats, this->numFeats));
    arma::mat  X = df.submat(r, c);
    this->supp = support(X, this->numFeats);
    this->partitionMap = new std::unordered_map<u_int, arma::vec>();
    this->leafRowMap   = new std::unordered_map<u_int, arma::uvec>();
    this->intervalStack = new std::vector<Interval*>();

    /* initialize default interval using the lb/ub of the support */
    arma::vec defaultInterval(2 * this->numFeats, arma::fill::zeros);
    /* populate the interval first with the support values */
    for (u_int i = 0; i < this->numFeats; i++) {
        defaultInterval(2*i)   = this->supp(i, 0);
        defaultInterval(2*i+1) = this->supp(i, 1);
    }
    this->defInterval = defaultInterval;

    this->root  = buildTreeSort(this->rowIds, this->treeSSE, this->numRows);
    
} // end Tree constructor

Node* Tree::buildTreeSort(arma::uvec rowIds, double currSSE, u_int nodesize) { 
    // note: this function is called every time a tree is built on 
    // a child node; each output will give the root node of the tree

    /* CHECK TERMINATING CONDITIONS: 
        (1) minsplit = 20; min # of observations that must exist in node
            in order for a split to be considered
        (2) minbucket = round(minsplit / 3); min number of observations
            in any leaf node -- this one might need to be checked when
            proposing splits (do this one later)
        (3) cp = 0.01; any split that does not decrease overall lack of 
            fit by a factor of cp is not attempted
        note: seems like (2) and (3) should be done when the split
        is being proposed, rather than when checking for terminating
        conditions at the beginning of the recursive call
    */ 
    
    // check terminating conditions to break out of recursive calls
    if (nodesize <= this->nodeMinLimit) { 
        Node* leaf = new Node(this->z.rows(rowIds), rowIds);
        // make copy of default interval
        arma::vec leafInterval = this->defInterval;
        /* populate the intervals in the partitionMap for this leaf */
        populateInterval(leafInterval);
        (*(this->partitionMap))[this->numLeaves] = leafInterval;
        (*(this->leafRowMap))[this->numLeaves] = leaf->getLeafRows();
        this->numLeaves++;
        return leaf; // return pointer to leaf Node
    }

    // else: not at leaf node, rather a decision  node
    /* iterate through features, rows (data) to find (1) optimal 
        splitting feat and (2) optimal splitting value for that feature
    */
    double minSSE  = std::numeric_limits<double>::infinity();

    double optThreshold; // optimal split value
    u_int  optFeature;   // optimal feature to use in the split (col #)
    double optLeftSSE, optRightSSE; // minSSE = optLeftSSE + optrightSSE

    arma::uvec left, right;
    arma::uvec optLeft, optRight;
    u_int numFeats = this->numFeats;
    u_int numRows  = this->numRows;
    double leftSSE, rightSSE, propSSE; // store the SSE values
    u_int leftsize, rightsize;

    // since features are 1-indexed and numFeats features, this loop 
    // should go from 1 to numFeats (inclusive)
    for (u_int d = 1; d <= numFeats; d++) { // loop thru feats/cols

        /* idea: Sort response, d-th feature, and rowId by the feauture.
                    Then, when we propose feature split values, we do not
                    have to compare values, O(n)), to find the left and
                    right splits, we can just group indices together based
                    on relative positions from the proposed split 
        */
        arma::uvec sortedIndex = getSortedIndex(this->z, rowIds, d);

        // construct the sortedIndex output consisting of rowIds in sorted order
        // by iterating through the d-th array of sorted indices and adding
        // elements of the map onto the output array if we encounter them
        // in the d-th array of sorted indices -- this way, we guarantee 
        // encountering the arrays in "sorted" order
        
        u_int start = this->minBucket; 
        u_int n = rowIds.n_elem;
        u_int end = n - this->minBucket + 1;
        /*  --- TODO: this is the loop that we want to parallelize  --*/
        for (u_int j = start; j < end; j++) { // this iterates thru rows

            arma::uvec leftrows  = arma::conv_to<arma::uvec>::from(
                arma::linspace(0, j-1, j));
            arma::uvec rightrows = arma::conv_to<arma::uvec>::from(
                arma::linspace(j, n-1, n-j));

            left  = sortedIndex.elem(leftrows);
            right = sortedIndex.elem(rightrows);
            double threshold = this->z(sortedIndex(j-1), d);

            leftSSE  = calculateSSE(left);
            rightSSE = calculateSSE(right);
            propSSE  = leftSSE + rightSSE;
            
            if (propSSE < minSSE) {
                /* these individual SSE values are later used to 
                    compute the cp value to determine if the split is 
                    worth making */
                minSSE = propSSE; // TODO: rename to avoid confusion
                optLeftSSE = leftSSE;
                optRightSSE = rightSSE; 
                optThreshold = threshold;
                optFeature = d;
                optLeft = left;    // left row indices
                optRight = right;  // right row indices
                leftsize = j;
                rightsize = n-j;
            }

        } // end inner for() -- iter through rows to find optimal split

    } // end for() over features ------ end OUTER FOR LOOP()

    double cpProp  = calcSplitCp(currSSE, optLeftSSE, optRightSSE); 
    // Rcpp::Rcout << "cp value = " << cpProp << std::endl;
    
    /* ---------------- CHECK TERMINATING CONDITIONS -----------------*/ 
    /* shouldn't need to check the bucket condittion b/c construction of
        left/right nodes takes this into account already */
    // bool CONDITION_BUCKET = optLeft.size() < minBucket || 
    //                         optRight.size() < minBucket;
    bool CONDITION_CP = cpProp < this->cp;
    if (CONDITION_CP) {
        // check the terminating condition 
        Node* leaf = new Node(this->z.rows(rowIds), rowIds);
        arma::vec leafInterval = this->defInterval;
        /* populate the intervals in the partitionMap for this leaf */
        populateInterval(leafInterval);
        (*(this->partitionMap))[this->numLeaves] = leafInterval;
        (*(this->leafRowMap))[this->numLeaves] = leaf->getLeafRows();
        this->numLeaves++;
        return leaf; 
    } // end of if() checking terminating condition

    /* -------------------------------------------------------------- */
    
    /* ELSE: build the current node to have left/right split based on
                the optimal SSEs from the above calculations */ 

    // construct node using optimal value, column, data, left, right
    // TODO: correct the minSSE value to have it be the node SSE value
    // so that we can extract this in future iterations to save time
    Node* node = new Node(
        optThreshold, optFeature, z, optLeft, optRight
    );

    /* ------------------- start interval creation ------------------ */ 

    // obtain feature (this is just the column number)
    u_int feature = optFeature - 1; 
    double lb, ub, threshVal; 
    lb = this->supp(feature, 0);
    ub = this->supp(feature, 1);
    Interval* leftInterval = new Interval(lb, optThreshold, feature);
    Interval* rightInterval = new Interval(optThreshold, ub, feature);

    /* ---------------- end interval creation ----------------------- */

    // push left interval onto the interval stack
    (*(this->intervalStack)).push_back(leftInterval);
    node->left  = buildTreeSort(optLeft, optLeftSSE, leftsize);
    delete (*(this->intervalStack)).back();
    (*(this->intervalStack)).pop_back();

    // push right interval onto interval stack
    (*(this->intervalStack)).push_back(rightInterval);
    node->right = buildTreeSort(optRight, optRightSSE, rightsize);
    delete (*(this->intervalStack)).back();
    (*(this->intervalStack)).pop_back();

    return node;
} // end of buildTreeSort() function
/* ------------------------ END TREE BUILDING ALGO #1 ------------------------*/

/* --- GENERAL TREE CONSTRUCTOR WHICH CALLS DIFFERENT TREE BUILDING ALGO --- */
Tree::Tree(arma::mat df, bool fast) {

    this->z         = df;
    this->numRows   = df.n_rows;
    this->numFeats  = df.n_cols - 1;
    this->rowIds = arma::conv_to<arma::uvec>::from(
            arma::linspace(0, this->numRows-1, this->numRows));
    this->numLeaves = 0;
    this->numSplits = 0; 
    // this is the SSE for a tree with no splits
    this->treeSSE   = sse(df.col(0), numRows, arma::mean(df.col(0)));

    /* terminating condition variables */
    this->nodeMinLimit = 20; // min # obs in a node to consider a split
    this->minBucket    = nodeMinLimit / 3;
    this->cp           = 0.01;

    /* initialize variables related to extracting the partition */
    arma::uvec r = arma::conv_to<arma::uvec>::from(
            arma::linspace(0, this->numRows-1, this->numRows));
    arma::uvec c = arma::conv_to<arma::uvec>::from(
            arma::linspace(1, this->numFeats, this->numFeats));
    arma::mat  X = df.submat(r, c);
    this->supp = support(X, this->numFeats);
    this->partitionMap = new std::unordered_map<u_int, arma::vec>();
    this->leafRowMap   = new std::unordered_map<u_int, arma::uvec>();
    this->intervalStack = new std::vector<Interval*>();

    /* initialize default interval using the lb/ub of the support */
    arma::vec defaultInterval(2 * this->numFeats, arma::fill::zeros);
    /* populate the interval first with the support values */
    for (u_int i = 0; i < this->numFeats; i++) {
        defaultInterval(2*i)   = this->supp(i, 0);
        defaultInterval(2*i+1) = this->supp(i, 1);
    }
    this->defInterval = defaultInterval;

    /* up to here, all code above is the same between old and new code */

    if (fast) {
        /*  create the map that maps feature to vector of sorted indices based 
        on feature value -- for each of the (cts) features, find the indices
        that give the feature in ascending order */
        std::unordered_map<u_int, arma::uvec> map; 
        for (u_int d = 1; d <= this->numFeats; d++) {
            map[d] = getSortedIndex(this->z, this->rowIds, d);
        }
        this->featureIndexMap = map; 
        yloc.reserve(this->numRows);
        for (u_int i = 0; i < this->numRows; i++) {
            yloc[i] = 1; // by default, all are in the root node (1st node)
        }
        // TODO: replace the following line with the updated buildtree algo
        this->root = fastbuild(1, 0, this->numRows, this->rowIds);
    } else {
        this->root  = buildTreeSort(this->rowIds, this->treeSSE, this->numRows);
    }
    
} // end Tree constructor that uses map implementation


/* ----------------------- START TREE BUILDING ALGO #3 -----------------------*/
// TODO: implement the tree building algorithm that uses the optimization
// done in the rpart() source code
Node* Tree::fastbuild(u_int nodenum, u_int start, u_int end, arma::uvec rowIds) {

    // Rcpp::Rcout << "node: " << nodenum << std::endl;

    u_int n = end - start; // number of observations in (this) current node
    if (n <= this->nodeMinLimit) {
        // Rcpp::Rcout << "not enough obs to consider a split" << std::endl;
        Node* leaf = new Node(this->z.rows(rowIds), rowIds);
        // make copy of default interval
        arma::vec leafInterval = this->defInterval;
        /* populate the intervals in the partitionMap for this leaf */
        populateInterval(leafInterval);
        (*(this->partitionMap))[this->numLeaves] = leafInterval;
        (*(this->leafRowMap))[this->numLeaves] = leaf->getLeafRows();
        this->numLeaves++;
        return leaf; // return pointer to leaf Node
    }

    // Rcpp::Rcout << "checked node min limit" << std::endl;

    double cpProp; 
    /* --------------- find optimal splitting feature + value --------------- */
    /* this routine will give us: 
     *      nleft, nright
     *      optimal start, end value which will be used to make recursive calls
     *      best improvement (cp) value for the split
     */
    u_int kk, k; // kk is the index coming from the sorted indices
    arma::vec y(n, arma::fill::zeros);
    arma::vec x(n, arma::fill::zeros);
    arma::vec x_star(n, arma::fill::zeros);
    double bestimprove = 0; // used to determine if we keep a split
    double improve;

    /* variables for the optimal split */
    u_int splitvar;  // splitvar \in [1, numFeats]
    double splitval; // value to partition the n data points
    u_int n_left_best, n_right_best; // # obs in left/right nodes in opt split

    double left_sum, right_sum;
    int n_left, n_right;
    double tmp;
    double ybar;

    for (u_int d = 1; d <= this->numFeats; d++) {
        /* extract the sorted indices for the d-th feature; sortedindex contains
         * numRows-many elements, but we only end up using (end - start) many
         * elements; this is where the algorithm bottlenecked in prev. versions
         */
        arma::uvec sortedindex = this->featureIndexMap[d]; 
        // extract elements start to end from sortedindex, y (response), X(,d)
        k = 0;
        for (u_int j = start; j < end; j++) { 
            kk = sortedindex(j);
            y(k) = (this->z)(kk, 0);
            x(k) = (this->z)(kk, d);
            k++;
        } // end iteration over the sorted indices -> y, x are constructed

        // Rcpp::Rcout << "populated y, x vectors" << std::endl;

        /* next: using (x, y) above, find the optimal split for THIS feature
         * this is the anova() / (rp_choose) function in the rpart() code 
         */
        n_left = 0;           // left node starts with no observations
        n_right = n;          // right node starts with all observations
        left_sum  = 0;        // nleft * (leftmean - xbar) = 0 b/c first iter
        right_sum = 0;        // (rightmean - xbar) = 0 when all in right node
        ybar = arma::mean(y); // grand mean
        /*
         * Note: we use the identity:
         * improvement = w_l * (leftmean  - grandmean)^2 + 
         *               w_r * (rightmean - grandmean)^2
         *             = (left_sum  - grandmean)^2 / n_left + 
         *               (right_sum - grandmean)^2 / n_right
         * improvement is iteratively computed by +/- tmp from terms that
         * contribute to the left/right nodes. 
         */
        for (u_int i = 0; n_right >= this->minBucket; i++) {
            n_left++;
            n_right--;
            tmp = (y(i) - ybar);
            left_sum += tmp;
            right_sum -= tmp;
            // can skip improvement calculation for the first nodeMinLimit - 1
            // iters b/c the left node will not have enough observations
            if (n_left >= this->minBucket) {
                improve = left_sum * left_sum / n_left + 
                    right_sum * right_sum / n_right;
                if (improve > bestimprove) {
                    bestimprove = improve;
                    splitvar = d;
                    x_star = x;             // save the SORTED feature vector
                    splitval = x(i);        // calculated differently in anova.c
                    n_left_best = n_left;   // calculated in nodesplit
                    n_right_best = n_right; // calculated in nodesplit  
                } // end if() checking for best improvement so far
            } // end if() checking if left node has enough observations
        } // end loop finding the best improvement for the d-th feature

    } // end iteration over features

    // Rcpp::Rcout << "optimal split val + feature found" << std::endl;

    /* -------------- optimal splitting feature + value found --------------- */


    /* ---- check if improvement is enough to actually perform the split ---- */ 
    cpProp = bestimprove / this->treeSSE;
    // Rcpp::Rcout << "cp value = " << cpProp << std::endl;
    if (cpProp < this->cp) {
        // not worth splitting, form a leaf and return
        Node* leaf = new Node(this->z.rows(rowIds), rowIds);
        arma::vec leafInterval = this->defInterval;
        /* populate the intervals in the partitionMap for this leaf */
        populateInterval(leafInterval);
        (*(this->partitionMap))[this->numLeaves] = leafInterval;
        (*(this->leafRowMap))[this->numLeaves] = leaf->getLeafRows();
        this->numLeaves++;
        return leaf; // return pointer to leaf Node
    }

    // Rcpp::Rcout << "checked for sufficient improvement" << std::endl;

    /* else: perform the left/right split and recursively build tree */

    // (1) prepare for the split --> nodesplit() func in rpart implementation
    // update the yloc vector ('which' vector in rpart)
    u_int j; 
    u_int leftchild = 2 * nodenum;
    u_int rightchild = leftchild + 1;

    

    arma::uvec leftrows  = arma::conv_to<arma::uvec>::from(
        arma::linspace(start, 
                       start + n_left_best - 1, 
                       n_left_best));

    arma::uvec rightrows = arma::conv_to<arma::uvec>::from(
        arma::linspace(start + n_left_best, 
                       start + n_left_best + n_right_best - 1, 
                       n_right_best));

    arma::uvec left  = (this->featureIndexMap[splitvar]).elem(leftrows);
    arma::uvec right = (this->featureIndexMap[splitvar]).elem(rightrows);

    // updated method to assign node for each observation
    for (u_int i = 0; i < left.n_elem; i++) {
        yloc[left(i)] = leftchild;
    }
    for (u_int i = 0; i < right.n_elem; i++) {
        yloc[right(i)] = rightchild;
    }

    // use original d-th feature to assign observations to left/right node
    // arma::uvec sortedindex = this->featureIndexMap[splitvar]; 
    // x = (this->z).col(splitvar); 
    // for (u_int i = start; i < end; i++) {
    //     j = sortedindex(i);
    //     if (x(j) <= splitval) {
    //         yloc[j] = leftchild;
    //     } else {
    //         yloc[j] = rightchild;
    //     }
    // }

    /* (2) update the featureIndexMap so that within the left/right gorups, 
     * the elements are sorted (see bottom of nodesplit.c)
     * logic explanation: we already know that within each column (feature), the
     * indices are sorted that their feature values are in ascending order. 
     * We split the target rows into two groups: left or right, and within each
     * group, we preserve the order of the indices so that within each group, 
     * they still give the features in ascending order
     */
    u_int i1, i2;
    arma::uvec sortedindex;
    for (u_int d = 1; d <= this->numFeats; d++) {
        if (d == splitvar) { 
            // don't need to update anything for splitvar since left/right are
            // already sorted based on how the data are partitioned
            continue;
        }
        sortedindex = this->featureIndexMap[d]; 
        // initialize temp vector to store the right node indices
        arma::uvec rightnode(this->numRows, arma::fill::zeros); 
        i1 = start;
        i2 = start + n_left_best;
        for (u_int i = start; i < end; i++) {
            j = sortedindex(i);
            if (yloc[j] == leftchild) {
                sortedindex(i1++) = sortedindex(i);
            } else if (yloc[j] == rightchild) {
                rightnode(i2++) = sortedindex(i);
            } else { 
                // in our version of the tree building algorithm, this case
                // shouldn't happen because all target rows will belong in
                // either the left or right node
                Rcpp::Rcout << "neither left nor right" << std::endl;
            }
        } // end inner for iterating over the feature values

        // replace the values for the right node 
        for (u_int i = start + n_left_best; i < end; i++) {
            sortedindex(i) = rightnode(i);
        }

        // update + replace the vector of indices in featureIndexMap
        this->featureIndexMap[d] = sortedindex;

    } // end outer for iterating over features in the map

    /* ------------------------- perform the split -------------------------- */ 

    Node* node = new Node(splitval, splitvar);

    /* ------------------- start interval creation ------------------ */ 
    // obtain feature (this is just the column number)
    u_int feature = splitvar - 1; 
    double lb, ub; 
    lb = this->supp(feature, 0);
    ub = this->supp(feature, 1);
    Interval* leftInterval = new Interval(lb, splitval, feature);
    Interval* rightInterval = new Interval(splitval, ub, feature);
    /* ---------------- end interval creation ----------------------- */

    /* recursive call on left node */ 
    // push left interval onto the interval stack
    (*(this->intervalStack)).push_back(leftInterval);
    node->left = fastbuild(leftchild, 
                           start, 
                           start + n_left_best,
                           left);
    delete (*(this->intervalStack)).back();
    (*(this->intervalStack)).pop_back();

    /* recursive call on right node */ 
    // push right interval onto interval stack
    (*(this->intervalStack)).push_back(rightInterval);
    node->right = fastbuild(rightchild, 
                            start + n_left_best, 
                            start + n_left_best + n_right_best,
                            right);
    delete (*(this->intervalStack)).back();
    (*(this->intervalStack)).pop_back();


    return node;
} // end of fastbuild() function


/* ------------------------ END TREE BUILDING ALGO #3 ------------------------*/





/* -------------- PRINT TREE DECISION NODES TO HELP WITH DEBUG -------------- */
void Tree::dfs(Node* ptr, std::string spacing) {
    Rcpp::Rcout << spacing;
    ptr->printNode();
    if (ptr->left) {
        Rcpp::Rcout << spacing << "---> true";
        dfs(ptr->left, spacing + " ");
    }
    if (ptr->right) {
        Rcpp::Rcout << spacing << "---> false";
        dfs(ptr->right, spacing + " ");
    }
}

void Tree::printTree() {
    dfs(this->root);
} 


/* -------------------------- HELPER FUNCTIONS -------------------------------*/

arma::uvec Tree::getSortedIndex(arma::mat data, arma::uvec rowvec, u_int d) {
    /*  sortOnFeatureSub(): returns the row indices on the scale of the 
        ORIGINAL row indices (1 to nrow(data)), rather than on the scale 
        of subset. This allows us to avoid creating a new dataset as we 
        go deeper into the tree, and instead just rely on keep8ing track 
        of indices that give us sorted versions of the original feature 
        when we subset (call .elem())

        data   : full dataset; N x (D+1), where response is in 0-th col
        rowvec : vector of row indices for the relevant node
        d      : the index of the feature (col of the data) to sort on
    */
    arma::vec x = data.col(d);        // extract the d-th feature col
    arma::vec xsub = x.elem(rowvec);  // extract relevant rows
    // obtain the indices that place the d-th feature in ascending order
    arma::uvec sortedRowIndex = arma::sort_index(xsub);
    // Rcpp::Rcout << xsub.elem(sortedRowIndex) << std::endl;
    // return ORIGINAL indices arranged based on order of d-th feature

    /*  dimension check: note that sortedRowIndex and rowvec and xsub have 
        the same # of elements. We are simply using the sorted (on xsub) indices
        to obtain the order in terms of the original rowvec indices
    */
    return rowvec.elem(sortedRowIndex); 
}


double Tree::calcImprove(arma::uvec left, arma::uvec right) {
    u_int n_left = left.n_elem;
    u_int n_right = right.n_elem;
    arma::vec y = this->z.col(0);
    arma::vec yleft = y.elem(left);
    arma::vec yRight = y.elem(right);
    double leftmean = arma::mean(yleft);
    double rightmean = arma::mean(yRight);
    double grandmean = n_left * leftmean + n_right * rightmean;

    return n_left * std::pow(leftmean - grandmean , 2) + 
        n_right * std::pow(rightmean - grandmean, 2);
}

void Tree::populateInterval(arma::vec& leafInterval) {
    /*  populateInterval() : takes the default leaf interval and 
        fills the entries with the intervals that are current in the
        intervalStack. Intervals from this stack represent decision
        nodes that we have traversed along while building the tree. 
        Note: to revert to old implementation, just paste the following 
        for loop under the leftInterval initialization in the 
        buildTree()  function and delete the populateInterval() 
        function call */ 
    for (const auto interval : *intervalStack) {
        double lb = interval->lb;
        double ub = interval->ub;
        unsigned int col = interval->feature; 
        leafInterval(2*col) = std::max(leafInterval(2*col), lb);
        leafInterval(2*col+1) = std::min(leafInterval(2*col+1), ub);
    }
} // end populateInterval() function


double Tree::calcSplitCp(double currSSE, double leftSSE, double rightSSE) {

    /*  calcSplitCp(): return the cp value for the proposed split. this
        will be used to evaluate if a split is worth doing. 

        leftSSE  : SSE of the left child
        rightSSE : SSE of the right child
        currSSE  : SSE of the current node (parent of left/right) if
                   there is no left/right split (if curr node is leaf)
    */ 
    return (currSSE - (leftSSE + rightSSE)) / this->treeSSE;
} // end calcSplitCp() function


double Tree::sse(arma::vec x, int n, double xbar) {
    double res = 0;
    for (u_int i = 0; i < n; i++) {
        res += std::pow(x(i)- xbar, 2);
    }
    return res;
} // end sse() function)


double Tree::calculateSSE(arma::uvec rowIndex) {
    arma::vec y = this->z.col(0); // response stored in column 0
    arma::vec ySub = y.elem(rowIndex);
    double ybar = arma::mean(ySub);
    return sse(ySub, ySub.n_elem, ybar);
} // end calculateSSE() function


arma::mat Tree::support(arma::mat X, u_int nfeats) {
    arma::mat s(nfeats, 2, arma::fill::zeros);
    for (u_int d = 0; d < nfeats; d++) {
        s(d, 0) = X.col(d).min();
        s(d, 1) = X.col(d).max();
    }
    return s;
} // end support() function






/*  ----------------------------- getters ----------------------------------- */
int Tree::getLeaves() {
    return this->numLeaves;
}
u_int Tree::getNumFeats() {
    return this->numFeats;
}
u_int Tree::getNumRows() { 
    return this->numRows;
}
double Tree::getSSE() {
    return this->treeSSE;
}
std::unordered_map<u_int, arma::vec>* Tree::getPartition() {
    return this->partitionMap;
}
std::unordered_map<u_int, arma::uvec>* Tree::getLeafRowMap() {
    return this->leafRowMap;
}
