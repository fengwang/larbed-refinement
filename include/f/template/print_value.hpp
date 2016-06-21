#ifndef MPRINT_VALUE_HPP_INCLUDED_SEDPFOINASDLFKJN4OIUANJDFSLKJAMNDFSIUHAJSFDOIU
#define MPRINT_VALUE_HPP_INCLUDED_SEDPFOINASDLFKJN4OIUANJDFSLKJAMNDFSIUHAJSFDOIU 

#include <f/template/type_at.hpp>

#include <cstddef>
#include <iostream>

namespace f
{

    struct print_value
    {
        template<typename Type>
        std::ostream& operator()( std::ostream& lhs, Type const& rhs ) const
        {
            return lhs << rhs;
        }

        template<typename Type, typename... Types>
        std::ostream& operator()( std::ostream& lhs, Type const& rhs, Types const& ... rhss ) const
        {
            lhs << rhs << ", ";
            return print_value()(lhs, rhss...);
        }
    };

}//namespace f

#endif//_PRINT_VALUE_HPP_INCLUDED_SEDPFOINASDLFKJN4OIUANJDFSLKJAMNDFSIUHAJSFDOIU 

