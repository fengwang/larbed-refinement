#ifndef MZEROS_HPP_INCLUDED_SOFD4IJ489USAFIJSFKLJXVCNSFDJSFDJKLHALSFKJHASOFIU498SDFAIJHORUISAIFDJOEIR
#define MZEROS_HPP_INCLUDED_SOFD4IJ489USAFIJSFKLJXVCNSFDJSFDJKLHALSFKJHASOFIU498SDFAIJHORUISAIFDJOEIR

#include <f/matrix/matrix.hpp>

#include <algorithm>
#include <cstddef>

namespace f
{
    template<typename T,
             std::size_t D = 256,
             typename A = std::allocator<typename remove_const<typename remove_reference<T>::result_type>::result_type> >
    matrix<T,D,A> const zeros( const std::size_t r, const std::size_t c )
    {
        matrix<T,D,A> ans{ r, c, T(0) };
        return ans;
    }

    template<typename T,
             std::size_t D = 256,
             typename A = std::allocator<typename remove_const<typename remove_reference<T>::result_type>::result_type> >
    matrix<T,D,A> const zeros( const std::size_t n )
    {
        return zeros<T,D,A>( n, n );
    }

    template<typename T, std::size_t D, typename A >
    matrix<T,D,A> const zeros( const matrix<T,D,A>& m )
    {
        return zeros<T,D,A>( m.row(), m.col() );
    }

    template<typename T, std::size_t D, typename A >
    matrix<T,D,A> const zeros( const matrix<T,D,A>&, const std::size_t r, const std::size_t c )
    {
        return zeros<T,D,A>( r, c );
    }

}//namespace f

#endif//_ZEROS_HPP_INCLUDED_SOFD4IJ489USAFIJSFKLJXVCNSFDJSFDJKLHALSFKJHASOFIU498SDFAIJHORUISAIFDJOEIR

