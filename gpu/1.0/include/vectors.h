#ifndef VECTORS_H_
#define VECTORS_H_

namespace GNNS {
	template <typename T>
	struct Vectors {
		unsigned num;
		unsigned dim;
		T* data;
	};
}

#endif