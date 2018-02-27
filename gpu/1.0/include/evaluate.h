#ifndef EVALUATE_H_
#define EVALUATE_H_

#include <Vectors.h>
#include <iostream>

namespace GNNS {
	float evaluate(Vectors<int> & result, Vectors<int> & truth, int truth_dim) {
		int num = result.num;
		int dim = result.dim;
		int count = 0;

		for (int i = 0; i < num; i++)
			for (int j = 0; j < dim; j++)
				if (result.data[i*dim + j] == truth.data[i*truth_dim + j])
					count++;

		return 1.0 * count / (num*dim);
	}
}

#endif