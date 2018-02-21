#ifndef DISTANCE_H_
#define DISTANCE_H_

#include <vector>
#include <utility>
#include <algorithm>
using std::vector;
using std::pair;

namespace GNNS {

    struct pair_less{
        bool operator()(pair<int, float> a, pair<int, float> b){
            return a.second < b.second;
        }
    };

    class Distance {
    protected:
        float quantity;
    public:
        float get(){ return quantity; }
    };

    template <typename T>
    class Euclidean : public Distance{

    public:
        Euclidean(vector<T> a, vector<T> b){
            quantity = calculate(a, b);
        }

    private:
        float calculate(vector<T> a, vector<T> b){
            typename vector<T>::const_iterator ai = a.cbegin();
            typename vector<T>::const_iterator bi = b.cbegin();
            
            float result = 0;
            for( ; ai != a.cend(); ai++, bi++)
                result += 1.0 * (*ai - *bi) * (*ai - *bi);
            return result;
        }
    };
}

#endif