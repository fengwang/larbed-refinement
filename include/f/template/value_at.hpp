#ifndef MVALUE_AT_HPP_INCLUDED_DSPOI498YFDSKLJHASFDKJNVAIDUFJASKFJD49UHFDS98YKJH
#define MVALUE_AT_HPP_INCLUDED_DSPOI498YFDSKLJHASFDKJNVAIDUFJASKFJD49UHFDS98YKJH

#include <f/template/type_at.hpp>

#include <cstddef>

namespace f
{

    template< std::size_t N, typename T, typename... Types >
    struct value_at
    {
        typedef typename type_at< N, T, Types...>::result_type result_type;

        result_type operator()( T, Types... vts ) const 
        {
            return value_at<N-1, Types...>()( vts... );
        }
    };

    template< typename T, typename... Types >
    struct value_at< 0, T, Types... >
    {
        typedef T result_type;

        result_type operator()( T vt, Types... ) const 
        {
            return vt;
        }
    };
    
}//namespace f

#endif//_VALUE_AT_HPP_INCLUDED_DSPOI498YFDSKLJHASFDKJNVAIDUFJASKFJD49UHFDS98YKJH

