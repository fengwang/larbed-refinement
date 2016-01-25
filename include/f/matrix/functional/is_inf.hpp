#ifndef MIS_INF_HPP_INCLUDED_SFDOJASFOIJSADOIJSDFO9YU8498YUASFKIHSFDUIOHVIUOHSAF978Y4UIJHASFUYH4EU
#define MIS_INF_HPP_INCLUDED_SFDOJASFOIJSADOIJSDFO9YU8498YUASFKIHSFDUIOHVIUOHSAF978Y4UIJHASFUYH4EU

#include <f/matrix/matrix.hpp>
#include <f/algorithm/for_each.hpp>

#include <cmath>

namespace f
{

    template< typename T, std::size_t N, typename A >
    matrix<bool> is_inf( const matrix<T,N,A>& m )
    {
        matrix<bool> ans( m.row(), m.col() );
        for_each( m.begin(), m.end(), ans.begin(), []( const T& v, bool& a ) { a = std::isinf( v ) ? true : false; } );
        return ans;
    }

    template< typename T, std::size_t N, typename A >
    matrix<bool> isinf( const matrix<T,N,A>& m )
    {
        return is_inf( m );
    }

}//namespace f

#endif//_IS_INF_HPP_INCLUDED_SFDOJASFOIJSADOIJSDFO9YU8498YUASFKIHSFDUIOHVIUOHSAF978Y4UIJHASFUYH4EU

