

#include "Tree.h"

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


    /* choose tree building routine: nonzero code is optimized */ 
    this->root  = buildTreeSort(this->rowIds, this->treeSSE, this->numRows);
    
} // end Tree constructor


Tree::Tree(arma::mat df, bool indicator) {

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

    /*  create the map that maps feature to vector of sorted indices based 
        on feature value    */
    std::unordered_map<u_int, arma::uvec> map; 
    for (u_int d = 1; d <= this->numFeats; d++) {
        map[d] = getSortedIndex(this->z, this->rowIds, d);
    }
    this->featureIndexMap = map; 
    // Rcpp::Rcout << "created map for sorted indices for each feature" << 
    //      std::endl;


    /* choose tree building routine: nonzero code is optimized */ 
    this->root = buildTreeMap(this->rowIds, this->treeSSE, this->numRows);
} // end Tree constructor that uses map implementation


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


// this version of build uses the hashmap of the row indices to avoid sorting
Node* Tree::buildTreeMap(arma::uvec rowIds, double currSSE, u_int nodesize) { 
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

    /* TODO: add all of rowIds into a map */ 
    std::unordered_map<u_int, bool> rowmap;
    for (u_int k = 0; k < nodesize; k++) {
        rowmap[rowIds(k)] = true;
    }

    // since features are 1-indexed and numFeats features, this loop 
    // should go from 1 to numFeats (inclusive)
    for (u_int d = 1; d <= numFeats; d++) { // loop thru feats/cols

        /* idea: Sort response, d-th feature, and rowId by the feauture.
                    Then, when we propose feature split values, we do not
                    have to compare values, O(n)), to find the left and
                    right splits, we can just group indices together based
                    on relative positions from the proposed split 
        */

        // arma::uvec sortedIndex = getSortedIndex(this->z, rowIds, d);
        // construct the sortedIndex output consisting of rowIds in sorted order
        // by iterating through the d-th array of sorted indices and adding
        // elements of the map onto the output array if we encounter them
        // in the d-th array of sorted indices -- this way, we guarantee 
        // encountering the arrays in "sorted" order
        arma::uvec sortedIndex(nodesize); 
        arma::uvec index_d = (this->featureIndexMap)[d]; // n-dim "sorted" indices
        u_int i = 0;
        for (u_int m = 0; m < this->numRows; m++) {
            if (rowmap.find(index_d(m)) != rowmap.end()) {
                // add index_d(j) to the sortedIndex vector
                sortedIndex(i) = index_d(m);
                i++;
                if (i == nodesize) {
                    break;
                }
            }
        }
        
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
    node->left  = buildTreeMap(optLeft, optLeftSSE, leftsize);
    delete (*(this->intervalStack)).back();
    (*(this->intervalStack)).pop_back();

    // push right interval onto interval stack
    (*(this->intervalStack)).push_back(rightInterval);
    node->right = buildTreeMap(optRight, optRightSSE, rightsize);
    delete (*(this->intervalStack)).back();
    (*(this->intervalStack)).pop_back();

    return node;
} // end of fasterBuildTree() function


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


double Tree::sse(arma::vec x, int n, double xbar) {
    double res = 0;
    for (u_int i = 0; i < n; i++) {
        res += std::pow(x(i)- xbar, 2);
    }
    return res;
} // end sse() function)


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


double Tree::calculateSSE(arma::uvec rowIndex) {
    arma::vec y = this->z.col(0); // response stored in column 0
    arma::vec ySub = y.elem(rowIndex);
    double ybar = arma::mean(ySub);
    return sse(ySub, ySub.n_elem, ybar);
} // end sse() function


// compute the SSE associated with this decision node (rule)
double Tree::calculateRuleSSE(arma::uvec leftRows, arma::uvec rightRows) {
    arma::vec y = this->z.col(0);
    // subset out y-values for left node
    arma::vec yLeft = y.elem(leftRows); 
    // subset out y-values for right node
    arma::vec yRight = y.elem(rightRows);
    // compute left-mean
    double leftMean = arma::mean(yLeft);
    // compute right-mean
    double rightMean = arma::mean(yRight);

    // Rcpp::Rcout<< "leftMean: " << leftMean << std::endl;
    // Rcpp::Rcout<< "rightMean: " << rightMean << std::endl;
    int nLeft = yLeft.n_elem;
    int nRight = yRight.n_elem;
    // compute sse for left
    double sseLeft = sse(yLeft, nLeft, leftMean);
    // compute sse for right
    double sseRight = sse(yRight, nRight, rightMean);
    // compute node sse
    // Rcpp::Rcout<< "sseLeft: " << sseLeft << std::endl;
    // Rcpp::Rcout<< "sseRight: " << sseRight << std::endl;
    return sseLeft + sseRight;
} // end calculateRuleSSE() function


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






/* ------------------------- old implementations ---------------------------- */

// Tree::Tree(arma::mat df) {
//     this->z         = df;
//     this->numRows   = df.n_rows;
//     this->numFeats  = df.n_cols - 1;
//     this->rowIds = arma::conv_to<arma::uvec>::from(
//             arma::linspace(0, this->numRows-1, this->numRows));
//     this->numLeaves = 0;
//     this->numSplits = 0; 
//     // this is the SSE for a tree with no splits
//     this->treeSSE   = sse(df.col(0), numRows, arma::mean(df.col(0)));

//     /* terminating condition variables */
//     this->nodeMinLimit = 20; // min # obs in a node to consider a split
//     this->minBucket    = nodeMinLimit / 3;
//     this->cp           = 0.01;
//     this->root      = buildTree(this->rowIds);

// } // end Tree() constructor

// Node* Tree::buildTree(arma::uvec rowIds) { 
//     // note: this function is called every time a tree is built on 
//     // a child node; each output will give the root node of the tree

//     /* CHECK TERMINATING CONDITIONS: 
//         (1) minsplit = 20; min # of observations that must exist in node
//             in order for a split to be considered
//         (2) minbucket = round(minsplit / 3); min number of observations
//             in any leaf node -- this one might need to be checked when
//             proposing splits (do this one later)
//         (3) cp = 0.01; any split that does not decrease overall lack of 
//            fit by a factor of cp is not attempted
//         note: seems like (2) and (3) should be done when the split
//         is being proposed, rather than when checking for terminating
//         conditions at the beginning of the recursive call
//     */ 

//     // int nodeMinLimit = 20;
//     // int minBucket = nodeMinLimit / 3; 
//     // double cp = 0.01; // min value of improvement to do a split
    
//     // check terminating conditions to break out of recursive calls
//     if (rowIds.n_elem <= this->nodeMinLimit) { 
//         this->numLeaves++;
//         Node* leaf = new Node(this->z.rows(rowIds), rowIds);
//         return leaf; // return pointer to leaf Node
//     }

//     // else: not at leaf node, rather a decision  node
//     /* iterate through features, rows (data) to find (1) optimal 
//        splitting feat and (2) optimal splitting value for that feature
//     */
//     double minSSE  = std::numeric_limits<double>::infinity();

//     double optThreshold; // optimal split value
//     u_int  optFeature;   // optimal feature to use in the split (col #)
//     double optLeftSSE, optRightSSE; // minSSE = optLeftSSE + optrightSSE

//     arma::uvec left, right;
//     arma::uvec optLeft, optRight;
//     u_int numFeats = this->numFeats;
//     u_int numRows  = this->numRows;
//     double leftSSE, rightSSE, propSSE; // store the SSE values

//     // since features are 1-indexed and numFeats features, this loop 
//     // should go from 1 to numFeats (inclusive)
//     for (u_int d = 1; d <= numFeats; d++) { // loop thru feats/cols
//         // start from col 1 b/c y in col 0
//         // for (u_int n = 0; n < numRows; n++) { // loop thru data

//         /* idea: Sort response, d-th feature, and rowId by the feauture.
//                  Then, when we propose feature split values, we do not
//                  have to compare values, O(n)), to find the left and
//                  right splits, we can just group indices together based
//                  on relative positions from the proposed split 
//         */

//         for (u_int n = 0; n < rowIds.n_elem; n++) {
//             u_int n_i = rowIds(n);
//             double threshold = z(n_i, d);
//             // iterate through the rows corresponding to feature d 
//             // to find the optimal value on which to partition (bisect) 
//             // the data propose X(d, n) as the splitting rule

//             // double threshold = z(n, d); // propose new split value
//             std::vector<u_int> rightRows, leftRows;

//             // construct the left/right row index vectors 
//             for (const auto &i : rowIds) {
//                 // compare val with each of the vals in other rows
//                 double val_i = z(i, d);
//                 if (val_i <= threshold) {
//                     // determine right child vs. left child membership 
//                     // based on value compared to threshold
//                     leftRows.push_back(i);
//                 } else {
//                     rightRows.push_back(i);
//                 }

//             } // end for() creating the left/right row index vectors

//             if (rightRows.size() <= minBucket || 
//                 leftRows.size() <= minBucket) {
//                 // we do this check because we already checked the 
//                 // condition of a leaf node in beginning of function
//                 continue; // go back to start of INNER for over the rows
//             }

//             // compute SSE associated with this decision rule
//             // convert rightRow, leftRow into arma::uvec
//             left  = arma::conv_to<arma::uvec>::from(leftRows);
//             right = arma::conv_to<arma::uvec>::from(rightRows);

//             leftSSE  = calculateSSE(left);
//             rightSSE = calculateSSE(right);
//             propSSE  = leftSSE + rightSSE;
//             // Rcpp::Rcout<< "left: " << left << std::endl;
//             // Rcpp::Rcout<< "right: " << right << std::endl;
//             // Rcpp::Rcout<< "sse: " << propSSE << std::endl;
            
//             if (propSSE < minSSE) {
//                 /* these individual SSE values are later used to 
//                    compute the cp value to determine if the split is 
//                    worth making */
//                 // minSSE is sum of left, right; NOT the node's SSE 
//                 minSSE = propSSE; // TODO: rename to avoid confusion
//                 optLeftSSE = leftSSE;
//                 optRightSSE = rightSSE; 
//                 // TODO: store threshold value; this defines partition
//                 optThreshold = threshold;
//                 // TODO: store col number b/c this gives partition #
//                 // don't need to store the row (n) b/c we only need
//                 // the value (threshold) and the feature (d) so we know
//                 // where to define this 1-d interval in the partition
//                 optFeature = d;
//                 optLeft = left;    // left row indices
//                 optRight = right;  // right row indices
//             } // end if() checking if we beat the current minimum

//         } // end for() over data, over z(i, d), for i in [0, nRows]

//     } // end for() over features ------ end OUTER FOR LOOP()

//     // DEBUG CODE TO LOOK AT VALUES:
//     // Rcpp::Rcout<< "optimal feature: " << optFeature << " ("
//     //     << optThreshold << ")" << std::endl;
//     // Rcpp::Rcout<< "optimal split val: " << optThreshold << std::endl;
//     // Rcpp::Rcout<< "min SSE: " << minSSE << std::endl;

//     /* compute cp value for the current node's proposed split */ 
//     // compute SSE of curr node if we do NOT perform optimal l/r splits
//     // TODO: can store this value in the node if we build the left/right
//     // to avoid recalculating 
//     // will match this->treeSSE on 1st call
//     double currSSE = calculateSSE(rowIds); 
//     double cpProp  = calcSplitCp(currSSE, optLeftSSE, optRightSSE); 
    
//     /* ---------------- CHECK TERMINATING CONDITIONS -----------------*/ 
//     bool CONDITION_BUCKET = optLeft.size() < minBucket || 
//                             optRight.size() < minBucket;
//     bool CONDITION_CP = cpProp < this->cp;
//     if (CONDITION_BUCKET || CONDITION_CP) {
//         // check the terminating condition - don't want any
//         // leaf nodes that have fewer than minBucket many points
//         this->numLeaves++;
//         Node* leaf = new Node(this->z.rows(rowIds), rowIds);
//         // if (CONDITION_BUCKET) {
//         //     Rcpp::Rcout<< "hit minbucket condition -> leaf value = " 
//         //         << leaf->getLeafVal() << std::endl;
//         // } else {
//         //     Rcpp::Rcout<< "hit cp condition -> leaf value = " <<
//         //         leaf->getLeafVal() << std::endl;
//         // }
//         return leaf; // return pointer to leaf Node
//     }

//     /* -------------------------------------------------------------- */
    
//     /* ELSE: build the current node to have left/right split based on
//              the optimal SSEs from the above calculations */ 

//     int leftCount = optLeft.n_elem;
//     int rightCount = optRight.n_elem;

//     // construct node using optimal value, column, data, left, right
//     Node* node = new Node(
//         optThreshold, optFeature, z, optLeft, optRight, minSSE
//     );
//     // Rcpp::Rcout << "# in left: " << node->leftCount << std::endl;
//     // Rcpp::Rcout << "# in right: " << node->rightCount << std::endl;
//     // Rcpp::Rcout<< "L: " << node->leftCount << " / R: " << 
//     //     node->rightCount << std::endl;
//     // Rcpp::Rcout<< "-----------------------------------" << std::endl;

//     // build left and right nodes
//     // in order to test one iteration of this function, just take out
//     // left, right subtree function calls
//     node->left  = buildTree(optLeft);
//     node->right = buildTree(optRight);

//     return node;
// } // end of buildTree() function





// void dfs(Node* ptr, std::string spacing = "") {
//     Rcpp::Rcout << spacing;
//     ptr->printNode();
//     if (ptr->left) {
//         Rcpp::Rcout << spacing << "---> true";
//         dfs(ptr->left, spacing + " ");
//     }
//     if (ptr->right) {
//         Rcpp::Rcout << spacing << "---> false";
//         dfs(ptr->right, spacing + " ");
//     }
// }

// void printTree() {
//     dfs(root);
// }

// void Tree::dfs(Node* node, unsigned int& k, std::vector<Interval*> intervalStack,
//     arma::mat& partition, arma::mat& supp, 
//     std::unordered_map<u_int, arma::uvec>& leafRowMap) {

//     /*  k:          current partition set that we are updating
//         decisions:  stack that holdes Interval objects
//         partition:  nLeaves x (2 * nFeats) matrix, storing intervals row-wise
//     */

//     if (node->isLeaf) {
//         /*
//             extract the interval's feature
//             replace the current interval values for that feature with the 
//             updated interval values: lb = max(curr, new), ub = min(curr, new);
//             this shrinks the interval to accommodate a tighter bound as we
//             go deeper into the tree
//         */
//         for (const auto interval : intervalStack) {
//             double lb = interval->lb;
//             double ub = interval->ub; 
//             unsigned int col = interval->feature; // features are 1-indexed

//             // need to take max, min to accommodate for shrinking intervals,
//             // further restrictions on the same feature as we go down the tree
//             // partition(k, 2*col) = std::max(partition(k, 2*col), lb);
//             // partition(k, 2*col+1) = std::min(partition(k, 2*col+1), ub);

//             partition(2*col, k) = std::max(partition(2*col, k), lb);
//             partition(2*col+1, k) = std::min(partition(2*col+1, k), ub);
//         } // end of for() updating the k-th leaf

//         // add the rows corresponding to this leaf into the hashmap
//         // this is used in the hybrid algo to find candidate/representative 
//         // points in each partition (leaf) set
//         // NOTE: since we are doing this in the same order as storing the
//         // intervals (above), we ensure that the k-th partition set
//         // corresponds to the leafRows that we store in the map below
//         leafRowMap[k] = node->getLeafRows();

//         // Rcpp::Rcout << "row " << k << " : leaf value = " << 
//         // node->getLeafVal() << std::endl;
//         k++; // move to next leaf node / interval to update
//         return;
//     } // end of terminating condition

//     /*  keep going down the tree and pushing the rules onto the stack until we
//         reach a leaf node, then we use all elements in the stack to construct
//         the interval. after constructing the interval, pop the leaf node off 
//         so we can continue down another path to find the next leaf node
//     */ 

//     // extract the rule for the curr node -> this will give left and ride nodes
//     // features are stored in the Node object as 1-index because the data 
//     // looks like [y|X], so features start from column 1 rather than column 0
//     // but when creating subsequent maps and matrices, we still want the 
//     // first feature to be in the 0-th (first) column (element in map)
//     unsigned int feature = node->getFeature() - 1;
//     double lb = supp(feature, 0);
//     double ub = supp(feature, 1);
//     double threshVal = node->getThreshold();  // threshold value for node
//     // construct interval for the left node
//     Interval* leftInterval = new Interval(lb, threshVal, feature);
//     // construct interval for the right node
//     Interval* rightInterval = new Interval(threshVal, ub, feature);

//     // travel down left node
//     intervalStack.push_back(leftInterval);
//     dfs(node->left, k, intervalStack, partition, supp, leafRowMap);
//     delete intervalStack.back();
//     intervalStack.pop_back();

//     // travel down right node
//     intervalStack.push_back(rightInterval);
//     dfs(node->right, k, intervalStack, partition, supp, leafRowMap);
//     delete intervalStack.back();
//     intervalStack.pop_back();

// } // end dfs() function
