#ifndef MSTREAM_OPERATOR_HPP_INCLUDED_SDPONASPOIJ3P9HASFDLH4P9HIASFDLHI4P9H8AUFS
#define MSTREAM_OPERATOR_HPP_INCLUDED_SDPONASPOIJ3P9HASFDLH4P9HIASFDLHI4P9H8AUFS

#include <f/matrix/details/crtp/typedef.hpp>
#include <iostream>
#include <algorithm>

namespace f
{
    template<typename Matrix, typename Type, typename Allocator>
    struct crtp_stream_operator
    {
        typedef Matrix                                                          zen_type;
        typedef crtp_typedef<Type, Allocator>                                   type_proxy_type;
        typedef typename type_proxy_type::size_type                             size_type;
        typedef typename type_proxy_type::value_type                            value_type;

        friend std::ostream& operator <<( std::ostream& lhs, zen_type const& rhs )
        {
            for ( size_type i = 0; i < rhs.row(); ++i )
            {
                std::copy( rhs.row_begin( i ), rhs.row_end( i ), std::ostream_iterator<value_type> ( lhs, " \t " ) );
                lhs << "\n";
            }
            return lhs;
        }


    };//struct

}

#endif//_STREAM_OPERATOR_HPP_INCLUDED_SDPONASPOIJ3P9HASFDLH4P9HIASFDLHI4P9H8AUFS

