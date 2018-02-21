#ifndef SERVE_H_
#define SERVE_H_

#include <distance.h>
#include <ctime>
#include <cstdlib>

namespace GNNS {
    template <typename T>
    vector<vector<int>> 
    serve(vector<vector<T>> base, 
    vector<vector<T>> query, 
    vector<vector<pair<int, float>>> kNN_Graph,
    int k, int r, int s, int e){
        pair_less cmp;
        int num = base.size();
        int query_size = query.size();
        vector<vector<int>> * result = new vector<vector<int>>();

        for ( int ri = 0; ri < query_size; ri ++){
            std::printf("%.2f %%\r", 100.0 * ri / query_size);
            vector<pair<int, float>> candidates;

            for ( int i = 0; i < r; i++ ){
                srand((unsigned)time(0));
                int index = rand() % num;

                for ( int j = 0; j < s; j++ ){
                    float dist;
                    float minDist = kNN_Graph.at(index).at(0).second;
                    float minIndex = kNN_Graph.at(index).at(0).first;
                    
                    for ( int h = 1;  h < e; h++ ){
                        dist = kNN_Graph.at(index).at(h).second;
                        if (dist < minDist){
                            minDist = dist;
                            minIndex = h;
                        }
                    }

                    index = minIndex;
                    pair<int, float> new_candidate = std::make_pair(index, 
                        Euclidean<float>(query.at(ri), base.at(index)).get());
                    candidates.push_back(new_candidate);
                }
            }

            std::sort(candidates.begin(), candidates.end(), cmp);
            vector<pair<int,float>>::iterator it = candidates.begin();
            int count = 0;
            int temp = (*it).first;
            for( ; it != candidates.end(); ){
                if ( count == 0 ){
                    count = 1;
                    continue;
                }
                else if ( count == k )
                    break;
                else if ( (*it).first == temp )
                    it = candidates.erase(it);
                else{
                    temp = (*it).first;
                    ++count;
                    ++it;
                }
            }

            vector<int> new_result;
            for( int i = 0; i < k; i++ )
                new_result.push_back(candidates.at(i).first);
            result->push_back(new_result);
        }

        std::cout << std::endl;
        return *result;
    }
}

#endif