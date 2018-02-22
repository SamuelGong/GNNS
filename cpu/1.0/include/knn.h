#ifndef KNN_H_
#define KNN_H_

#include <setting.h>
#include <distance.h>
#include <io.h>
#include <process.h>
#include <fstream>

namespace GNNS {
	template <typename T>
	vector<vector<pair<int, float>>> & build_kNN_Graph(vector<vector<T>> base, int k) {

		if (_access(GRAPH_FILE, 0) == 0) {
			return read_file<pair<int, float>>(GRAPH_FILE);
		}

		pair_less cmp;
		int num = base.size();
		float * dist_set = new float[num*num];
		vector<vector<pair<int, float>>> * result = new vector<vector<pair<int, float>>>();

		for (int i = 0; i < num; i++) {
			std::printf("%.2f %%\r", 100.0 * i / num);
			std::cout << std::flush;

			for (int j = i; j < num; j++)
				if (j == i)
					dist_set[i*num + j] == 0;
				else
					dist_set[i*num + j] = dist_set[j*num + i] = Euclidean<T>(base.at(i), base.at(j)).get();

			vector<pair<int, float>> pairs_for_a_vertex;
			for (int j = 0; j < num; j++) {
				pair<int, float> new_pair = std::make_pair(j, dist_set[i*num + j]);
				pairs_for_a_vertex.push_back(new_pair);
			}

			std::sort(pairs_for_a_vertex.begin(), pairs_for_a_vertex.end(), cmp);
			vector<pair<int, float>> neighbors_for_a_vertex;
			for (int j = 1; j <= k; j++) {
				neighbors_for_a_vertex.push_back(pairs_for_a_vertex.at(j));
			}
			result->push_back(neighbors_for_a_vertex);
		}

		std::cout << std::endl;
		write_file<pair<int, float>>(GRAPH_FILE, *result);
		return *result;
	}
}

#endif