#ifndef DSOJDLSKFJ498IJAFLDSJDFONVOAISFALSFKDJNZKJNSDVOIUH4KJFDSAKFSDJHASFIUH4FA 
#define DSOJDLSKFJ498IJAFLDSJDFONVOAISFALSFKDJNZKJNSDVOIUH4KJFDSAKFSDJHASFIUH4FA

#include <f/coefficient/coefficient.hpp>
#include <f/matrix/matrix.hpp>

#include <complex>
#include <cstddef>

namespace f
{

    template< typename T, typename Itor >
    std::complex<T> const make_coefficient_element( std::complex<T>const& t, Itor diag_begin, Itor diag_end, std::size_t row, std::size_t col )
    {
        std::size_t const n = std::distance( diag_begin, diag_end );

        assert( std::real(t) < T{ 1.0e-100} );
        assert( std::real(t) > T{-1.0e-100} );
        assert( std::imag(t) > T{0.0} );

        assert( row < n );
        assert( col < n );

        if ( row != col )
            return coefficient<T>{t, diag_begin, diag_end}( row, col );

        return std::exp( t * ( *(diag_begin+row) ) );
    }

   
    template< typename T, typename Itor >
    matrix<std::complex<T> > const make_coefficient_matrix( std::complex<T> const& t, Itor diag_begin, Itor diag_end )
    {
        unsigned long const n = std::distance( diag_begin, diag_end );

        matrix<std::complex<T> > S( n, n );

        for ( unsigned long r = 0; r != n; ++r )
            for ( unsigned long c = 0; c != n; ++c )
                S[r][c] = make_coefficient_element( t, diag_begin, diag_end, r, c );
            
        return S; 
    }
   
    template< typename T, typename Itor >
    matrix<std::complex<T> > const make_coefficient_matrix( std::complex<T> const& t, Itor diag_begin, Itor diag_end, const unsigned long c )
    {
        unsigned long const n = std::distance( diag_begin, diag_end );

        matrix<std::complex<T> > S( n, 1 );

        for ( unsigned long r = 0; r != n; ++r )
                S[r][0] = make_coefficient_element( t, diag_begin, diag_end, r, c );
            
        return S; 
    }
   
}//namespace f

#endif//DSOJDLSKFJ498IJAFLDSJDFONVOAISFALSFKDJNZKJNSDVOIUH4KJFDSAKFSDJHASFIUH4FA

