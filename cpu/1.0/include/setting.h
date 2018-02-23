#ifndef SETTING_H_
#define SETTING_H_

#include <iostream>
#include <fstream>
#include <strstream>
#include <string>
using std::string;

int K_;
int K;
int R;
int S;
int E;
string BASE_FILE;
string QUERY_FILE;
string GROUND_FILE;
string GRAPH_FILE;
const char* CONFIG_FILE = "./config.txt";

namespace GNNS {
	void setting() {

		FILE *fp;
		if ((fp = fopen(CONFIG_FILE, "r")) == nullptr) {
			std::cout << "File open error!" << std::endl;
			exit(-1);
		}

		string parameters[9];
		for (int i = 0; i < 9; i++) {
			char ch = 'a';
			string temp;
			while (ch != ':')
				fread(&ch, sizeof(char), 1, fp);
			fread(&ch, sizeof(char), 1, fp);
			while (ch != '\n') {
				temp += ch;
				fread(&ch, sizeof(char), 1, fp);
			}
			parameters[i] = temp;
		}

		BASE_FILE = parameters[0];
		QUERY_FILE = parameters[1];
		GROUND_FILE = parameters[2];
		GRAPH_FILE = parameters[3];
		K_ = std::stoi(parameters[4]);
		K = std::stoi(parameters[5]);
		R = std::stoi(parameters[6]);
		S = std::stoi(parameters[7]);
		E = std::stoi(parameters[8]);

		fclose(fp);
	}
}

#endif