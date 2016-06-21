#ifndef KCASEOHHUJMLXFOYKJJUGLFOCCQUUWVBGLSUVGABCYLBUKDOXSLQMCOIUNAIRBHTGNYWKEVVL
#define KCASEOHHUJMLXFOYKJJUGLFOCCQUUWVBGLSUVGABCYLBUKDOXSLQMCOIUNAIRBHTGNYWKEVVL

#include <sstream>

namespace f
{

    template <typename T, typename U>
    T const lexical_cast( U const& from )
    {
        T var;

        std::stringstream ss;
        ss << from;
        ss >> var;

        return var;
    }

}//namespace f

#endif//KCASEOHHUJMLXFOYKJJUGLFOCCQUUWVBGLSUVGABCYLBUKDOXSLQMCOIUNAIRBHTGNYWKEVVL

