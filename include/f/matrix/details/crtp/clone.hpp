#ifndef MCLONE_HPP_INCLUDED_DFSPJ3POIJHAF98YH4KLJBNFVKLJAFD984YHKJJHSFAD9IUH4FSD
#define MCLONE_HPP_INCLUDED_DFSPJ3POIJHAF98YH4KLJBNFVKLJAFD984YHKJJHSFAD9IUH4FSD

#include <f/matrix/details/crtp/typedef.hpp>

#include <cassert>
#include <algorithm>

namespace f
{
    template<typename Matrix, typename Type, typename Allocator>
    struct crtp_clone
    {
        typedef Matrix                                                          zen_type;
        typedef crtp_typedef<Type, Allocator>                                   type_proxy_type;
        typedef typename type_proxy_type::size_type                             size_type;

        template<typename Other_Matrix>
        zen_type& clone( const Other_Matrix& other, size_type const r0, size_type const r1, size_type const c0, size_type const c1 )
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            assert( r1 > r0 );
            assert( c1 > c0 );
            zen.resize( r1 - r0, c1 - c0 );
            for ( size_type i = r0; i != r1; ++i )
            { std::copy( other.row_begin( i ) + c0, other.row_begin( i ) + c1, zen.row_begin( i - r0 ) ); }
            return zen;
        }

    };//struct crtp_clone

}

#endif//_CLONE_HPP_INCLUDED_DFSPJ3POIJHAF98YH4KLJBNFVKLJAFD984YHKJJHSFAD9IUH4FSD

