#ifndef MIS_ORTHOGONAL_HPP_INCLUDED_WHATOIJSFOIUSFDOI4U98UASFLKHOWOAJABOIUSKLJSODIJCENDIJSFI49IFSDF
#define MIS_ORTHOGONAL_HPP_INCLUDED_WHATOIJSFOIUSFDOI4U98UASFLKHOWOAJABOIUSKLJSODIJCENDIJSFI49IFSDF

#include <f/matrix/matrix.hpp>

#include <algorithm>
#include <cstddef>
#include <cassert>

namespace f
{

    template< typename T, std::size_t N, typename A, typename F >
    bool is_orthogonal( const matrix<T,N,A>& m, F f )
    {
        if ( m.row() != m.col() ) return false;

        auto mm = m.transpose() * m;

        std::for_each( mm.diag_begin(), mm.diag_end(), [](T& v){ v -= T(1); } );

        return std::all_of( mm.begin(), mm.end(), f );
    }

    template< typename T, std::size_t N, typename A >
    bool is_orthogonal( const matrix<T,N,A>& m )
    {
        return is_orthogonal( m, [](const T v){ return v == T(0); } );
    }
}//namespace f

#endif//_IS_ORTHOGONAL_HPP_INCLUDED_WHATOIJSFOIUSFDOI4U98UASFLKHOWOAJABOIUSKLJSODIJCENDIJSFI49IFSDF

