#ifndef CFAQICXTTBSMLXORLABILTIFLFRIWVMIVLYHAGHGNVSKFQQTMNMBQNGTJNQACRMNMWSASXBQM
#define CFAQICXTTBSMLXORLABILTIFLFRIWVMIVLYHAGHGNVSKFQQTMNMBQNGTJNQACRMNMWSASXBQM

#include <algorithm> //for std::max std::min
#include <iostream>

namespace f
{
    struct marker_pair
    {
        typedef long value_type;
        value_type first;   //also the min
        value_type last;    //also the max

        marker_pair( const value_type first_, const value_type last_ ) : first( std::min( first_, last_ ) ), last( std::max( first_, last_ ) ) {}
    };

    bool operator == ( marker_pair const& lhs, marker_pair const& rhs )
    {
        if ( lhs.first == rhs.first && lhs.last == rhs.last ) return true;
        return false;
    }

    bool operator != ( marker_pair const& lhs, marker_pair const& rhs )
    {
        return !( lhs == rhs );
    }

    bool operator < ( marker_pair const& lhs, marker_pair const& rhs )
    {
        if ( lhs.first < rhs.first ) return true;
        if ( lhs.first > rhs.first ) return false;
        if ( lhs.last < rhs.last ) return true;
        return false;
    }

    bool operator <= ( marker_pair const& lhs, marker_pair const& rhs )
    {
        return ( lhs == rhs || lhs < rhs );
    }

    bool operator > ( marker_pair const& lhs, marker_pair const& rhs )
    {
        if ( lhs.first > rhs.first ) return true;
        if ( lhs.first < rhs.first ) return false;
        if ( lhs.last > rhs.last ) return true;
        return false;
    }

    bool operator >= ( marker_pair const& lhs, marker_pair const& rhs )
    {
        return ( lhs == rhs || lhs > rhs );
    }

    std::ostream& operator << ( std::ostream& os, marker_pair const& mp )
    {
        return os << "(" << mp.first << ", " << mp.last << ")";
    }

}//namespace f

#endif//CFAQICXTTBSMLXORLABILTIFLFRIWVMIVLYHAGHGNVSKFQQTMNMBQNGTJNQACRMNMWSASXBQM
