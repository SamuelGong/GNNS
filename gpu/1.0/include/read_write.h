#ifndef READ_WRITE_H_
#define READ_WRITE_H_

#include <vectors.h>
#include <string>
#include <iostream>
using std::string;

namespace GNNS {

	template <typename T>
	Vectors<T> & read_file(string path) {
		FILE *fp;

		if ((fp = fopen(path.c_str(), "rb")) == nullptr) {
			std::cout << "File open error!" << std::endl;
			exit(-1);
		}

		unsigned dim;
		unsigned file_size;
		fread(&dim, sizeof(int), 1, fp);
		fseek(fp, 0L, SEEK_END);
		file_size = ftell(fp);
		fseek(fp, 0L, SEEK_SET);

		Vectors<T> * result = new Vectors<T>();
		result->dim = dim;
		result->num = file_size / ((dim + 1) * sizeof(float));
		result->data = new T[result->num * result->dim];

		fseek(fp, 0L, SEEK_SET);
		unsigned count = 0;
		while (true) {
			fread(&dim, sizeof(int), 1, fp);
			if (feof(fp))
				break;
			fread(result->data + count * dim, sizeof(T), dim, fp);
			count ++;
		}

		fclose(fp);
		return *result;
	}

	template <typename T>
	void write_file(string path, const Vectors<T> & source) {
		FILE *fp;
		if ((fp = fopen(path.c_str(), "wb")) == nullptr) {
			std::cout << "File open error!" << std::endl;
			exit(-1);
		}

		int dim = source.dim;
		int num = source.num;
		for (int i = 0; i < num; i++) {
			fwrite(&dim, sizeof(int), 1, fp);
			fwrite(source.data + i * dim, sizeof(T), dim, fp);
		}
		
		fclose(fp);
	}
}

#endif
