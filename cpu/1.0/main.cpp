#include <read_write.h>
#include <setting.h>
#include <knn.h>
#include <serve.h>
#include <evaluate.h>
#include <iostream>

using namespace GNNS;

int main(){
    
    // read the file
    std::cout << "Reading base file..." << std::endl;
    vector<vector<float>> base = read_file<float>(BASE_FILE);
    std::cout << "Reading query file..." << std::endl;
    vector<vector<float>> query = read_file<float>(QUERY_FILE);

    // build a kNN graph
    std::cout << "Building kNN Graph..." << std::endl;
    vector<vector<pair<int, float>>> kNN_Graph = build_kNN_Graph<float>(base, K_);

    // serve the queries
    std::cout << "Serving the queries..." << std::endl;
    vector<vector<int>> result = serve<float>(base, query, kNN_Graph, K, R, S, E);

    // evaluate the result
    std::cout << "Reading ground_truth file..." << std::endl;
    vector<vector<int>> ground = read_file<int>(GROUND_FILE);
    std::cout << "Evaluating the result..." << std::endl;
    float prec = evaluate(result, ground);
    std::cout << "Precision: " << prec << std::endl;

    return 0;
}