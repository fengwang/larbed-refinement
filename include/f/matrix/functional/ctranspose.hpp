#ifndef MCTRANSPOSE_HPP_INCLUDED_SDIOJ4EIJFJIEIORJOFIWJVKLJJKSFDNVNNNNNNNNNNNNNNNNNNNNNNNNSFDKJHASJDO
#define MCTRANSPOSE_HPP_INCLUDED_SDIOJ4EIJFJIEIORJOFIWJVKLJJKSFDNVNNNNNNNNNNNNNNNNNNNNNNNNSFDKJHASJDO

#include <f/matrix/matrix.hpp>
#include <f/matrix/numeric/math.hpp>

#include <algorithm>
#include <cstddef>

namespace f
{
    template<typename T, std::size_t D, typename A>
    matrix<std::complex<T>,D,A> const ctranspose( const matrix<std::complex<T>,D,A>& m )
    {
        return conj( m.transpose() );
    }

}//namespace f

#endif//_CTRANSPOSE_HPP_INCLUDED_SDIOJ4EIJFJIEIORJOFIWJVKLJJKSFDNVNNNNNNNNNNNNNNNNNNNNNNNNSFDKJHASJDO

