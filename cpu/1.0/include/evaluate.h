#ifndef EVALUATE_H_
#define EVALUATE_H_

#include <vector>
using std::vector;

namespace GNNS {
	float evaluate(vector<vector<int>> & result, vector<vector<int>> & truth) {
		int num = result.size();
		int dim = result.at(0).size();
		int count = 0;

		for (int i = 0; i < num; i++)
			for (int j = 0; j < dim; j++)
				if (result.at(i).at(j) == truth.at(i).at(j))
					count++;

		return 1.0 * count / (num*dim);
	}
}

#endif