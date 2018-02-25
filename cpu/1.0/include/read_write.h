#ifndef READ_WRITE_H_
#define READ_WRITE_H_

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <io.h>
#include <process.h>

using std::vector;
using std::string;

namespace GNNS {
	template <typename T>
	vector<vector<T>> & read_file(string path) {

		FILE *fp;
		if ((fp = fopen(path.c_str(), "rb")) == nullptr) {
			std::cout << "File open error!" << std::endl;
			exit(-1);
		}

		T temp;
		int dim;
		vector<vector<T>> * result = new vector<vector<T>>();
		
		while (true) {
			vector<T> element; 
			fread(&dim, sizeof(int), 1, fp);
			if (feof(fp))
				break;

			for (int j = 0; j < dim; j++) {
				fread(&temp, sizeof(T), 1, fp);
				element.push_back(temp);
			}
			result->push_back(element);
		}
		
		fclose(fp);
		return *result;
	}

	template <typename T>
	void write_file(string path, const vector<vector<T>> & source) {
		FILE *fp;
		if ((fp = fopen(path.c_str(), "wb")) == nullptr) {
			std::cout << "File open error!" << std::endl;
			exit(-1);
		}

		int dim;
		int num = source.size();

		for (int i = 0; i < num; i++) {
			dim = source.at(i).size();
			fwrite(&dim, sizeof(int), 1, fp);
			for (int j = 0; j < dim; j++)
				fwrite(&(source.at(i).at(j)), sizeof(T), 1, fp);
		}
			
		fclose(fp);
	}

}

#endif