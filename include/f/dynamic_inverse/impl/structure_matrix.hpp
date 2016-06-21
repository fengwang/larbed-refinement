#ifndef HBODEAKUKUXBIEVACMUEUAJIJHJEWIEXUSGPCHUNNETHLYVYCDVFSWYIFVVNWCAWHSHASWEDJ
#define HBODEAKUKUXBIEVACMUEUAJIJHJEWIEXUSGPCHUNNETHLYVYCDVFSWYIFVVNWCAWHSHASWEDJ

#include <f/matrix/matrix.hpp>
#include <f/algorithm/for_each.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cassert>
#include <iterator>

namespace f
{

    template< typename T, typename Itor >
    matrix<std::complex<T> > const make_structure_matrix( matrix<std::size_t> const& ar, matrix<std::complex<T> > const& ug, Itor diag_begin, Itor diag_end )
    {
        assert( ar.row() );
        assert( ar.row() == ar.col() );
        assert( ug.size() );
        assert( ug.size() > *std::max_element( ar.begin(), ar.end() ) );
        assert( ar.row() == static_cast<std::size_t>(std::distance( diag_begin, diag_end )) );

        matrix<std::complex<T> > A( ar.row(), ar.col() );

        for ( std::size_t r = 0; r != A.row(); ++r )
            for ( std::size_t c = 0; c != A.col(); ++c )
                A[r][c] = ug[ar[r][c]][0];

        for_each( A.diag_begin(), A.diag_end(), diag_begin, []( std::complex<T>& a, T const& d ) { a = std::complex<T>{d, 0}; } ); 

        return A;
    }

    template< typename T, typename Itor >
    matrix<std::complex<T> > const make_structure_matrix( matrix<std::size_t> const& ar, matrix<T> const& ug, Itor diag_begin, Itor diag_end )
    {
        matrix<std::complex<T>> complex_ug( ug.row(), ug.col() );
        std::transform( ug.begin(), ug.end(), complex_ug.begin(), []( T const x ) { return std::complex<T>{ x, 0 }; } );
        return make_structure_matrix( ar, complex_ug, diag_begin, diag_end );
    }

}//namespace f

#endif//HBODEAKUKUXBIEVACMUEUAJIJHJEWIEXUSGPCHUNNETHLYVYCDVFSWYIFVVNWCAWHSHASWEDJ

