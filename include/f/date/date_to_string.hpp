#ifndef GNYWCIIBVBRRVHBYDRXBWTFICRLCHNISXHVENKFIPBSSKELGDYNLQASMOQBWTWVNKQYCVWNWO
#define GNYWCIIBVBRRVHBYDRXBWTFICRLCHNISXHVENKFIPBSSKELGDYNLQASMOQBWTWVNKQYCVWNWO

#include <iostream>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <sstream>
#include <cctype>
#include <algorithm>

namespace f
{

    inline std::string const date_to_string()
    {
        std::time_t rawtime;
        std::time( &rawtime );
        auto timeinfo = std::localtime( &rawtime );
        std::string ans{ std::asctime (timeinfo) };
        ans.resize( ans.size() - 1 );
        std::for_each( ans.begin(), ans.end(), []( char& ch ) { if( !isalnum(ch) ) ch = '_'; } );
        return ans;
    }

}//namespace f

#endif//GNYWCIIBVBRRVHBYDRXBWTFICRLCHNISXHVENKFIPBSSKELGDYNLQASMOQBWTWVNKQYCVWNWO

