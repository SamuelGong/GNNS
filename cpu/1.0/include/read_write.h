#ifndef READ_WRITE_H_
#define READ_WRITE_H_

#include <vector>
#include <fstream>
#include <iostream>
#include <io.h>
#include <process.h>

using std::vector;

namespace GNNS {
	void le_read(long buffer, size_t size, size_t count, FILE * stream) {
		for (int i = 0; i < count; i++)
			for (int j = 0; j < size; j++)
				fread((void*)(buffer + i*size + j), 1, 1, stream);
	} 

	void le_write(const long buffer, size_t size, size_t count, FILE * stream) {
		for (int i = 0; i < count; i++)
			for (int j = 0; j < size; j++)
				fwrite((const void*)(buffer + i*size + j), 1, 1, stream);
	}

	template <typename T>
	vector<vector<T>> & read_file(const char * path) {

		FILE *fp;
		if ((fp = fopen(path, "rb")) == nullptr) {
			std::cout << "File open error!" << std::endl;
			exit(-1);
		}

		T temp;
		int dim;
		vector<vector<T>> * result = new vector<vector<T>>();
		
		int flag = 0;
		while (true) {
			vector<T> element; 
			le_read(long(&dim), sizeof(int), 1, fp);
			if (feof(fp))
				break;

			for (int j = 0; j < dim; j++) {
				le_read(long(&temp), sizeof(T), 1, fp);
				element.push_back(temp);
			}
			result->push_back(element);
		}
		
		fclose(fp);
		return *result;
	}

	template <typename T>
	void write_file(const char* path, const vector<vector<T>> & source) {
		FILE *fp;
		if ((fp = fopen(path, "wb")) == nullptr) {
			std::cout << "File open error!" << std::endl;
			exit(-1);
		}

		int dim;
		int num = source.size();

		for (int i = 0; i < num; i++) {
			dim = source.at(i).size();
			le_write(long(&dim), sizeof(int), 1, fp);
			for (int j = 0; j < dim; j++)
				le_write(long(&(source.at(i).at(j))), sizeof(T), 1, fp);
		}
	}

}

#endif