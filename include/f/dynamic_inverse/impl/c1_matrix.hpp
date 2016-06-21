#ifndef SNMSOQMFJBOKDXYQAHXDOBLKIBEYGAEXNTCLEXBRACHHYHTKVCHJLHGUARPDLNDQAMCVXQVPL
#define SNMSOQMFJBOKDXYQAHXDOBLKIBEYGAEXNTCLEXBRACHHYHTKVCHJLHGUARPDLNDQAMCVXQVPL

#include <f/matrix/matrix.hpp>
#include <f/coefficient/coefficient.hpp>

#include <iterator>
#include <iostream>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cassert>

namespace f
{

    template<typename T>
    struct c1_data
    {
        std::size_t         index;          //0 for free variable
        std::complex<T>     coefficient;

        c1_data( std::size_t index_ = 0, std::complex<T> const& coefficient_ = std::complex<T> {0, 0} ) : index( index_ ), coefficient( coefficient_ ) {}

        c1_data& operator *= ( T const& val )
        {
            coefficient *= val;
            return *this;
        }

        c1_data& operator *= ( std::complex<T> const& com )
        {
            coefficient *= com;
            return *this;
        }
    };

    template<typename T>
    std::ostream& operator << ( std::ostream& os, c1_data<T> const& rhs )
    {
        return os << "[" << rhs.index << "--" << rhs.coefficient << "]";
    }

    template<typename T, typename Input_Itor>
    matrix<c1_data<T>> const make_c1_matrix( std::complex<T> const& thickness, Input_Itor diag_begin, Input_Itor diag_end, matrix<std::size_t> const& Ar )
    {
        std::size_t const n = std::distance( diag_begin, diag_end );
        assert( Ar.row() == Ar.col() );
        assert( n == Ar.row() );
        matrix<c1_data<T>> ans { n, n };
        coefficient<T> const coef
        {
            thickness, diag_begin, diag_end
        };

        for ( std::size_t r = 0; r != n; ++r )
        {
            for ( std::size_t c = 0; c != n; ++c )
                ans[r][c] = c1_data<T> { Ar[r][c], coef( r, c ) };
            ans[r][r] = c1_data<T> { 0, std::exp( thickness * ( *( diag_begin + r ) ) ) };
        }

        return ans;
    }

    template<typename T, typename Input_Itor>
    matrix<c1_data<T>> const make_c1_matrix( T thickness, Input_Itor diag_begin, Input_Itor diag_end, matrix<std::size_t> const& Ar )
    {
        return make_c1_matrix( std::complex<T> {0, thickness}, diag_begin, diag_end, Ar );
    }

}//namespace f

#endif//SNMSOQMFJBOKDXYQAHXDOBLKIBEYGAEXNTCLEXBRACHHYHTKVCHJLHGUARPDLNDQAMCVXQVPL

