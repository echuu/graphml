


// don't put into tree yet because we won't be able to easily test the functions
// without instantiating a tree object which we don't have a full picture of yet











// IMMEDIATE TODOS: 
// (1) integrate this into the Tree class so that we can instantiate a tree
//     and test this function (update: can do this without a tree object)
// (2) try with output cbind(y, X) from the R file and see the output after one
//     call to buildTree() -- note, function does not buildTree; it only
//     goes through one iteration of the tree fitting process
// (3) figure out how we want to store the rules in each node; ideally we keep
//     as is to avoid instantiating objects everywhere. at the end, since we are
//     returning a Node, we can take the things (stored in form of ints, doubles)
//     and throw them into a Node constructor (this helps us avoid using multi-arg)
//     return values (but if we modularize the part of the code that nald put
//     into a function, then we may need multi-arg return functions anyway, use
//     pointer in this case then)


