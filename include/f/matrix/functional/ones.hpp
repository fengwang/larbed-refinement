#ifndef MONES_HPP_INCLUDED_SOFD4IJ489USAFIJSFKLJXVCNSFDJSFDJKLHALSFKJHASOFIU498SDFAIJHORUISAIFDJOEIR
#define MONES_HPP_INCLUDED_SOFD4IJ489USAFIJSFKLJXVCNSFDJSFDJKLHALSFKJHASOFIU498SDFAIJHORUISAIFDJOEIR

#include <f/matrix/matrix.hpp>

#include <algorithm>
#include <cstddef>

namespace f
{
    template<typename T,
             std::size_t D = 256,
             typename A = std::allocator<typename remove_const<typename remove_reference<T>::result_type>::result_type> >
    matrix<T,D,A> const ones( const std::size_t r, const std::size_t c )
    {
        matrix<T> ans{ r, c, T(1) };
        return ans;
    }

    template<typename T,
             std::size_t D = 256,
             typename A = std::allocator<typename remove_const<typename remove_reference<T>::result_type>::result_type> >
    matrix<T,D,A> const ones( const std::size_t n )
    {
        return ones<T,D,A>( n, n );
    }

    template<typename T,
             std::size_t D = 256,
             typename A = std::allocator<typename remove_const<typename remove_reference<T>::result_type>::result_type> >
    matrix<T,D,A> const ones( const matrix<T,D,A>& m )
    {
        return ones<T,D,A>( m.row(), m.col() );
    }

}//namespace f

#endif//_ONES_HPP_INCLUDED_SOFD4IJ489USAFIJSFKLJXVCNSFDJSFDJKLHALSFKJHASOFIU498SDFAIJHORUISAIFDJOEIR

