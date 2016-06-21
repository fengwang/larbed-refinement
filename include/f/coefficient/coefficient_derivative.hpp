#ifndef XQAVOWQYUOKPIRAVPMIAHEXQWRANHDNOOAJYDGNMDLVKHWKNWSRNULYJCALOTKKHLYTXKCLBP
#define XQAVOWQYUOKPIRAVPMIAHEXQWRANHDNOOAJYDGNMDLVKHWKNWSRNULYJCALOTKKHLYTXKCLBP

#include <f/matrix/matrix.hpp>
#include <f/coefficient/expm.hpp>
#include <f/derivative/derivative.hpp>

#include <cassert>
#include <complex>
#include <cstddef>
#include <vector>

namespace f
{
    // I ---- the value type, float or double or long double
    // E ---- the derived class, impls the coefficient
    template< typename T, typename E >
    struct coefficient_derivative
    {
        typedef T                           value_type;
        typedef std::size_t                 size_type;
        typedef std::complex<value_type>    complex_type;
        typedef std::vector<value_type>     array_type;
        typedef f::matrix<complex_type>     complex_matrix_type;

        typedef E                           expm_impl_type;

        complex_matrix_type&                A;
        complex_type const&                 t;
        size_type                           column;
        array_type                          I;    

        template< typename Itor >
        coefficient_derivative( complex_matrix_type& A_, complex_type const& t_, size_type c_, Itor i_first, Itor i_last ): A( A_ ), t( t_ ), column( c_ ), I( i_first, i_last )
        {
            assert( A.col() > column );
            assert( I.size() = A.row() );
        }
        
        value_type operator()( const size_type r ) const
        {
            assert( r < A.row() );

            auto const& zen = static_cast<expm_impl_type const&>( *this );

            auto const& f = [this, &]( value_type x )  -> value_type
            { 
                auto const & A_rc_r = ((*this).A)[r][c].real(); //record
                (((*this).A)[r][(*this).column]).real(x);                       //modify

                auto const& S_rc = zen.expm( (*this).A, (*this).t, r, (*this).c ); // calculate S
                value_type const I_sim = S_rc.norm();
                value_type const diff = I_sim - ((*this).I)[r];

                (((*this).A)[r][c]).real(A_rc_r);                 //restore

                return diff*diff;
            };

            auto const& df = make_derivative<0>( f );

            return df( A[r][column] );
        }

    };

    template< typename T >
    struct coefficient_derivative_c1 : coefficient_derivative< T, coefficient_derivative_c1<T> >
    {
        typedef coefficient_derivative< T, coefficient_derivative_c1<T> >       host_type;
        typedef T                                                               value_type;
        typedef std::size_t                                                     size_type;
        typedef std::complex<value_type>                                        complex_type;
        typedef std::vector<value_type>                                         array_type;
        typedef f::matrix<complex_type>                                         complex_matrix_type;
        
        template< typename Itor >
        coefficient_derivative_c1( complex_matrix_type& A_, complex_type const& t_, size_type c_, Itor i_first, Itor i_last ): host_type( A_, t_, c_, i_first, i_last ) {}

        complex_type const expm( complex_matrix_type const& A, complex_type const& t, size_type r, size_type c ) const
        {
            return expm( A, t, r, c );
        }
    };

    template< typename T >
    struct coefficient_derivative_c2 : coefficient_derivative< T, coefficient_derivative_c2<T> >
    {
        typedef coefficient_derivative< T, coefficient_derivative_c2<T> >       host_type;
        typedef T                                                               value_type;
        typedef std::size_t                                                     size_type;
        typedef std::complex<value_type>                                        complex_type;
        typedef std::vector<value_type>                                         array_type;
        typedef f::matrix<complex_type>                                         complex_matrix_type;

        template< typename Itor >
        coefficient_derivative_c2( complex_matrix_type& A_, complex_type const& t_, size_type c_, Itor i_first, Itor i_last ): host_type( A_, t_, c_, i_first, i_last ) {}

        complex_type const expm( complex_matrix_type const& A, complex_type const& t, size_type r,size_type c ) const
        {
            return expm_2( A, t, r, c );
        }
    };

}//namespace f

#endif//XQAVOWQYUOKPIRAVPMIAHEXQWRANHDNOOAJYDGNMDLVKHWKNWSRNULYJCALOTKKHLYTXKCLBP
