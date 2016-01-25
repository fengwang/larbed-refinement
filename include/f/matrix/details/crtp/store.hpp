#ifndef MSTORE_HPP_INCLUDED_SDPOIASD9UHWEJHASKLJHAS94EYHALSFUKHSAIOUFHSFDAKLHSAF
#define MSTORE_HPP_INCLUDED_SDPOIASD9UHWEJHASKLJHAS94EYHALSFUKHSAIOUFHSFDAKLHSAF

#include <f/matrix/details/crtp/typedef.hpp>

namespace f
{
    template<typename Matrix, typename Type, typename Allocator>
    struct crtp_store
    {
        typedef Matrix                                                          zen_type;
        typedef crtp_typedef<Type, Allocator>                                   type_proxy_type;

        // TODO:
        //compress and store
        bool store( const char* const file_name ) const
        {
            return true;
        }

        // TODO:
        //restore from a compressed file
        bool restore( const char* const file_name )
        {
            return true;
        }

    };//struct crtp_store

}

#endif//_STORE_HPP_INCLUDED_SDPOIASD9UHWEJHASKLJHAS94EYHALSFUKHSAIOUFHSFDAKLHSAF

