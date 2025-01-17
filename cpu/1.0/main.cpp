#include <read_write.h>
#include <setting.h>
#include <knn.h>
#include <serve.h>
#include <evaluate.h>
#include <iostream>
#include <ctime>

using namespace GNNS;

int main() {

	time_t start, end;
	time_t start_all, end_all;
	
	// read the file
	start_all = clock();
	std::cout << "Reading the settings..." << std::endl;
	setting();
	std::cout << "Reading base file..." << std::endl;
	vector<vector<float>> base = read_file<float>(BASE_FILE);
	
	std::cout << "Reading query file..." << std::endl;
	vector<vector<float>> query = read_file<float>(QUERY_FILE);

	// build a kNN graph
	std::cout << "Building kNN Graph..." << std::endl;
	vector<vector<int>> kNN_Graph = build_kNN_Graph<float>(base, K_);

	// serve the queries
	std::cout << "Serving the queries..." << std::endl;
	start = clock();
	vector<vector<int>> result = serve<float>(base, query, kNN_Graph, K, R, S, E);
	end = clock();

	// evaluate the result
	std::cout << "Reading ground_truth file..." << std::endl;
	vector<vector<int>> ground = read_file<int>(GROUND_FILE);
	std::cout << "Evaluating the result..." << std::endl;
	float prec = evaluate(result, ground);
	end_all = clock();
	std::cout << "Precision: " << prec * 100.0 << "%" << std::endl;
	std::cout << "Time for serving the queries: " 
		<< (double)(end - start) / CLOCKS_PER_SEC << " second" << std::endl;
	std::cout << "Time for the overall process: "
		<< (double)(end_all - start_all) / CLOCKS_PER_SEC << " second" << std::endl;

	getchar();

	return 0;
}