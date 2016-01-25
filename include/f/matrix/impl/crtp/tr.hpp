#ifndef MTR_HPP_INCLUDED_SFDOPJISFDALKJ4O9SFDALSKFJ0389YUAFSLJH4E0HAFSDFAOIDFSFA
#define MTR_HPP_INCLUDED_SFDOPJISFDALKJ4O9SFDALSKFJ0389YUAFSLJH4E0HAFSDFAOIDFSFA

#include <f/matrix/impl/crtp/typedef.hpp>

#include <numeric>

namespace f
{
    template<typename Matrix, typename Type, std::size_t Default, typename Allocator>
    struct crtp_tr
    {
        typedef Matrix                                                          zen_type;
        typedef crtp_typedef<Type, Default, Allocator>                          type_proxy_type;
        typedef typename type_proxy_type::value_type                            value_type;

        const value_type tr() const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return std::accumulate( zen.diag_begin(), zen.diag_end(), value_type() );
        }

    };//struct

}

#endif//_TR_HPP_INCLUDED_SFDOPJISFDALKJ4O9SFDALSKFJ0389YUAFSLJH4E0HAFSDFAOIDFSFA

