#ifndef JCGETNWMVLGPOAHEYXPUPDNYXKJOANBMHLAIBSEONBSNARARXFAIIPQWUTEKMDIHXEWAYOEGY
#define JCGETNWMVLGPOAHEYXPUPDNYXKJOANBMHLAIBSEONBSNARARXFAIIPQWUTEKMDIHXEWAYOEGY

#include <f/coefficient/coefficient.hpp>
#include <f/matrix/matrix.hpp>

#include <complex>
#include <iterator>

namespace f
{
   
    template< typename T, typename A_Itor, typename Diag_Itor >
    matrix<T> const intensity( T const thickness, A_Itor col_begin, A_Itor col_end, Diag_Itor diag_begin, Diag_Itor diag_end )
    {
        assert( std::distance( col_begin, cole_end ) == std::distance( diag_begin, diag_end ) );

        coefficient<T> const coef( std::complex<T>{0.0, thickness}, diag_begin, diag_end );

        unsigned long const row = std::disntance( col_begin, col_end );

        matrix<T> ans{ row, 1 };

        ans[0][0] = std::norm( std::exp( thickness * (*col_begin) ) );

        for ( unsigned long r = 1; r != row; ++r )
            ans[r][0] = std::norm( (*(col_begin+r)) * coef( r, 0 ) );

        return ans;
    }

}//namespace f

#endif//JCGETNWMVLGPOAHEYXPUPDNYXKJOANBMHLAIBSEONBSNARARXFAIIPQWUTEKMDIHXEWAYOEGY

