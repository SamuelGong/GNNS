#ifndef READ_WRITE_H_
#define READ_WRITE_H_

#include <vector>
#include <fstream>
#include <iostream>

using std::vector;

namespace GNNS {
    template <typename T>
    vector<vector<T>> & read_file(const char* path){

        FILE *fp;
        if ((fp = fopen(path, "rb")) == nullptr){
            std::cout << "File open error!" << std::endl;
            exit(-1);
        }

        T temp;
        int dim;
        vector<vector<T>> * result = new vector<vector<T>>();

        while(!feof(fp)){
            vector<T> element;
            fread(&dim, sizeof(int), 1, fp);

            for(int j = 0; j < dim; j++){
                fread(&temp, sizeof(T), 1, fp);
                element.push_back(temp);
            }
            result->push_back(element);
        }

        fclose(fp);
        return *result;
    }
    
}

#endif