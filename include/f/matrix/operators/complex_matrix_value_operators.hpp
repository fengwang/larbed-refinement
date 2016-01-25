#ifndef NTPKCELBVXUUCCFPYCNSFXADHJPHSMMJTDRXTUIRWYDPFNRHADWUABXDYMSDKPSEBUQMVQXRC
#define NTPKCELBVXUUCCFPYCNSFXADHJPHSMMJTDRXTUIRWYDPFNRHADWUABXDYMSDKPSEBUQMVQXRC

#include <f/matrix/matrix.hpp>
#include <f/matrix/functional.hpp>

#include <cassert>
#include <cstddef>
#include <algorithm>
#include <complex>

namespace f
{

template< typename T, std::size_t D, typename A>
const matrix<std::complex<T>,D,A>
operator + ( const matrix<std::complex<T>,D,A>& lhs, const T& rhs )
{
    matrix<std::complex<T>,D,A> ans( lhs );
    std::transform( ans.begin(), ans.end(), ans.begin(), [rhs]( std::complex<T> const& x ) { return rhs+x; } );
    return ans;
}

template< typename T, std::size_t D, typename A>
const matrix<std::complex<T>,D,A>
operator + ( const T& lhs, const matrix<std::complex<T>,D,A>& rhs )
{
	return rhs + lhs;
}

template< typename T, std::size_t D, typename A>
const matrix<std::complex<T>,D,A>
operator - ( const matrix<std::complex<T>,D,A>& lhs, const T& rhs )
{
    matrix<std::complex<T>,D,A> ans( lhs );
    std::transform( ans.begin(), ans.end(), ans.begin(), [rhs]( std::complex<T> const& x ) { return x-rhs; } );
    return ans;
}

template< typename T, std::size_t D, typename A>
const matrix<std::complex<T>,D,A>
operator - ( const T& lhs, const matrix<std::complex<T>,D,A>& rhs )
{
    matrix<std::complex<T>,D,A> ans( rhs );
    std::transform( ans.begin(), ans.end(), ans.begin(), [lhs]( std::complex<T> const& x ) { return -lhs+x; } );
    return ans;
}

template< typename T, std::size_t D, typename A>
const matrix<std::complex<T>,D,A>
operator * ( const matrix<std::complex<T>,D,A>& lhs, const T& rhs )
{
	matrix<std::complex<T>,D,A> ans( lhs );
    std::transform( ans.begin(), ans.end(), ans.begin(), [rhs]( std::complex<T> const& x ) { return x*rhs; } );
	return ans;
}

template< typename T, std::size_t D, typename A>
const matrix<std::complex<T>,D,A>
operator * ( const T& lhs, const matrix<std::complex<T>,D,A>& rhs )
{
	return rhs * lhs;
}

template< typename T, std::size_t D, typename A>
const matrix<std::complex<T>,D,A>
operator / ( const matrix<std::complex<T>,D,A>& lhs, const T& rhs )
{
	matrix<std::complex<T>,D,A> ans( lhs );
    std::transform( ans.begin(), ans.end(), ans.begin(), [rhs]( std::complex<T> const& x ) { return x/rhs; } );
	return ans;
}
template< typename T, std::size_t D, typename A>
const matrix<std::complex<T>,D,A>
operator / ( const T& lhs, const matrix<std::complex<T>,D,A>& rhs )
{
	return lhs * rhs.inverse();
}

}//namespace f

#endif//NTPKCELBVXUUCCFPYCNSFXADHJPHSMMJTDRXTUIRWYDPFNRHADWUABXDYMSDKPSEBUQMVQXRC

