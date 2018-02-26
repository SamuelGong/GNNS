#ifndef SERVE_H_
#define SERVE_H_

#include <distance.h>
#include <ctime>
#include <cstdlib>

namespace GNNS {
	template <typename T>
	vector<vector<int>>
		serve(vector<vector<T>> & base,
			vector<vector<T>> & query,
			vector<vector<int>> & kNN_Graph,
			int k, int r, int s, int e) {
		pair_less cmp;
		int num = base.size();
		int query_size = query.size();
		vector<vector<int>> * result = new vector<vector<int>>();

		srand((unsigned)time(0));
		for (int ri = 0; ri < query_size; ri++) {
			std::printf("%.2f %%\r", 100.0 * ri / query_size);
			std::cout << std::flush;

			vector<pair<int, float>> candidates;
			for (int i = 0; i < r; i++) {
				int index = rand() % num;

				for (int j = 0; j < s; j++) {
					float tmp_dist;
					float tmp_index;
					float minIndex = kNN_Graph.at(index).at(0);
					float minDist = Euclidean<float>(query.at(ri),
						base.at(minIndex)).get();

					for (int h = 1; h < e; h++) {
						tmp_index = kNN_Graph.at(index).at(h);
						tmp_dist = Euclidean<float>(query.at(ri),
							base.at(tmp_index)).get();
						pair<int, float> new_candidate = std::make_pair(tmp_index, tmp_dist);
						candidates.push_back(new_candidate);
						if (tmp_dist < minDist) {
							minDist = tmp_dist;
							minIndex = tmp_index;
						}
					}

					index = minIndex;
				}
			}

			std::sort(candidates.begin(), candidates.end(), cmp);
			vector<pair<int, float>>::iterator it = candidates.begin();

			int count = 0;
			int temp = (*it).first;

			for (; it != candidates.end(); ) {
				if (count == 0) {
					++count;
					++it;
					continue;
				}
				else if (count == k)
					break;
				else if ((*it).first == temp)
					it = candidates.erase(it);
				else {
					temp = (*it).first;
					++count;
					++it;
				}
			}
			// std::cout << candidates.size() << std::endl;

			vector<int> new_result;
			for (int i = 0; i < k; i++) {
				new_result.push_back(candidates.at(i).first);
			}
			result->push_back(new_result);
		}

		std::cout << std::endl;
		return *result;
	}
}

#endif