#ifndef MRESIZE_HPP_INCLUDED_FDPONSF2398ASDFKLV9U8YH3KLJAFSD893YHALFIJHCVNJSIUFH
#define MRESIZE_HPP_INCLUDED_FDPONSF2398ASDFKLV9U8YH3KLJAFSD893YHALFIJHCVNJSIUFH

#include <f/matrix/details/crtp/typedef.hpp>

#include <algorithm>

namespace f
{
    template<typename Matrix, typename Type, typename Allocator>
    struct crtp_resize
    {
        typedef Matrix                                                          zen_type;
        typedef crtp_typedef<Type, Allocator>                          type_proxy_type;
        typedef typename type_proxy_type::size_type                             size_type;
        typedef typename type_proxy_type::value_type                            value_type;

        zen_type& resize( const size_type new_row, const size_type new_col, const value_type v = value_type( 0 ) )
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            if ( ( zen.row() == new_row ) && ( zen.col() == new_col ) )
            { return zen; }
            zen_type ans( new_row, new_col, v );
            const size_type the_row_to_copy = std::min( zen.row(), new_row );
            const size_type the_col_to_copy = std::min( zen.col(), new_col );
            for ( size_type i = 0; i < the_row_to_copy; ++i )
            { std::copy( zen.row_begin( i ), zen.row_begin( i ) + the_col_to_copy, ans.row_begin( i ) ); }
            zen.swap( ans );
            return zen;
        }

    };//struct crtp_resize

}

#endif//_RESIZE_HPP_INCLUDED_FDPONSF2398ASDFKLV9U8YH3KLJAFSD893YHALFIJHCVNJSIUFH

