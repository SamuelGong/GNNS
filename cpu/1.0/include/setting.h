#ifndef PARAM_H_
#define PARAM_H_

const char* BASE_FILE       = "../../../data/sift/siftsmall/siftsmall_base.fvecs";
const char* QUERY_FILE      = "../../../data/sift/siftsmall/siftsmall_query.fvecs";
const char* GROUND_FILE     = "../../../data/sift/siftsmall/siftsmall_groundtruth.ivecs";
const char* GRAPH_FILE      = "./graph";
const int K_    = 10;       // k for k-NN graph
const int K     = 20;       // the numeber of required nearest neighbors
const int R     = 20;       // the number of ramdon starts
const int S     = 20;       // the number of greedy steps
const int E     = 5;        // < k, the number of expansions

#endif