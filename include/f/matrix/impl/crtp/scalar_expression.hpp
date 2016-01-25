#ifndef MSCALAR_EXPRESSION_HPP_INCLUDED_SDPOIJSAD0LJKASFD09J498YAFSKVHNH9Y84EOIH
#define MSCALAR_EXPRESSION_HPP_INCLUDED_SDPOIJSAD0LJKASFD09J498YAFSKVHNH9Y84EOIH

#include <f/matrix/impl/crtp/typedef.hpp>
#include <f/matrix/impl/crtp/expression.hpp>

namespace f
{
    template<typename Value_Type, typename Type, std::size_t Default, typename Allocator>
    struct crtp_scalar_expression : public crtp_expression< crtp_scalar_expression<Value_Type, Type, Default, Allocator>, Type, Default, Allocator >
    {
        typedef crtp_typedef<Type, Default, Allocator>        type_proxy_type;
        typedef typename type_proxy_type::value_type          value_type;
        typedef typename type_proxy_type::size_type           size_type;

        Value_type const& value;

        crtp_matrix_value_plus_expression( const Value_Type& v ) : value( v ) {}

        value_type operator()( const size_type, const size_type ) const
        {
            return value;
        }

    };//struct

}

#endif//_SCALAR_EXPRESSION_HPP_INCLUDED_SDPOIJSAD0LJKASFD09J498YAFSKVHNH9Y84EOIH 

