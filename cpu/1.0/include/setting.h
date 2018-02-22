#ifndef PARAM_H_
#define PARAM_H_

const char* BASE_FILE       = "../../data/sift/siftsmall/siftsmall_base.fvecs";
const char* QUERY_FILE      = "../../data/sift/siftsmall/siftsmall_query.fvecs";
const char* GROUND_FILE     = "../../data/sift/siftsmall/siftsmall_groundtruth.ivecs";
const char* GRAPH_FILE      = "./bin/graph";
const int K_    = 1000;         // k for k-NN graph
const int K     = 10;           // the number of required nearest neighbors
const int R     = 1;            // the number of ramdon starts
const int S     = 8;            // the number of greedy steps
const int E     = 1000;         // < k, the number of expansions

#endif