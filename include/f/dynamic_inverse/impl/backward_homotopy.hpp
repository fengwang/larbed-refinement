#ifndef MBACKWARD_HOMOTOPY_HPP_INCLUDED_FADSOJASDKLJN3IU9HASDFKLJVAMBNVKJAHBSDFD
#define MBACKWARD_HOMOTOPY_HPP_INCLUDED_FADSOJASDKLJN3IU9HASDFKLJVAMBNVKJAHBSDFD

#include <f/dynamic_inverse/impl/scattering_matrix.hpp>
#include <f/dynamic_inverse/impl/double_square_coef_matrix.hpp>
#include <f/dynamic_inverse/impl/double_square_ug_fitting.hpp>
#include <f/matrix/matrix.hpp>
#include <f/matrix/numeric/expm.hpp>
#include <f/coefficient/coefficient_matrix.hpp>

#include <complex>
#include <cassert>
#include <cstddef>
#include <algorithm>
#include <vector>

namespace f
{

#if 0
        | Expm_00 Expm_01 ...... Expm_0n | | C1 0 |   | I_0 |
        | Expm_10 Expm_11 ...... Expm_1n | | C1 1 |   | I_1 |
        | .............................. | | .... | = | ... |
        | .............................. | | .... |   | ... |
        | Expm_n0 Expm_n1 ...... Expm_nn | | C1 n |   | I_n |
#endif
    //for one tilt ar matrix only
    template< typename T = double >
    T make_backward_homotopy(   matrix<std::size_t> const&  ar,                      // beam index matrix
                                matrix<T> const&            diag,                    // row major
                                matrix<T> const&            intensity,               // row major
                                matrix<T> const&            ug,                      // column major initial guess
                                std::complex<T> const&      lhs_thickness,           // C1 approximation part
                                std::complex<T> const&      rhs_thickness,           // expm part
                                matrix<T>&                  new_ug,                  // fitting result
                                std::size_t                 column_index = 0 )       // 
    {
        //dimension concept check
        assert( ar.row() );
        assert( ar.row() == ar.col() );
        assert( ar.row() == diag.col() );
        assert( diag.row() == intensity.row() );
        assert( diag.row() );
        assert( diag.col() == intensity.col() );
        assert( ug.col() == 1 );
        assert( ug.size() > *std::max_element( ar.begin(), ar.end() ) );
        assert( std::imag( lhs_thickness) > T{0} );
        assert( std::imag( rhs_thickness) > T{0} );
        assert( std::abs(std::real( lhs_thickness)) < T{1.0e-10} );
        assert( std::abs(std::real( rhs_thickness)) < T{1.0e-10} );
        assert( column_index < ar.row() );

        typedef T                                       value_type;
        typedef std::size_t                             size_type;
        typedef std::complex<value_type>                complex_type;
        typedef matrix<value_type>                      matrix_type;
        typedef matrix<complex_type>                    complex_matrix_type;

        size_type const n = ar.row();                   //dimension of Ar
        size_type const tilt = diag.row();              //expemental tilts
        size_type const m = ug.size();                  //unknowns in ug

        matrix_type x{ n*tilt, m+m+2, value_type{0} };

        for ( size_type diag_index = 0; diag_index != tilt; ++diag_index ) // foreach tilt
        { 
            auto const& lhs_s = make_scattering_matrix( ar, ug, diag.row_begin(diag_index), diag.row_end(diag_index), lhs_thickness );       //constant matrix
            auto const& rhs_s = make_coefficient_matrix( rhs_thickness, diag.row_begin(diag_index), diag.row_end(diag_index), column_index );//symbol column matrix

            for ( size_type row_index = 0; row_index != n; ++row_index )
            {
                size_type const row_index_in_x = diag_index * n + row_index;
                for ( size_type col_index = 0; col_index != n; ++col_index )
                {
                    complex_type const& lhs_complex_coef = lhs_s[row_index][col_index];
                    complex_type const& rhs_complex_coef = rhs_s[col_index][0];
                    complex_type const coef_product      = lhs_complex_coef * rhs_complex_coef;
                    value_type const real_product        = std::real( coef_product );
                    value_type const imag_product        = std::imag( coef_product );
                    size_type const ug_index             = ar[col_index][column_index];
                    if ( row_index == col_index ) // DC
                    {
                        x[row_index_in_x][0]  = real_product;
                        x[row_index_in_x][m+1] = imag_product;
                        //x[row_index_in_x][n] = imag_product;
                    }
                    else
                    {
                        x[row_index_in_x][ug_index+1]   = real_product;
                        x[row_index_in_x][ug_index+2+m] = imag_product;
                        //x[row_index_in_x][row_index]   = real_product;
                        //x[row_index_in_x][row_index] = imag_product;
                    }
                }//col_index
            }//row_index
        }//diag_index

        matrix_type y{ intensity };
        y.reshape( n*tilt, 1 );

        double const residual = double_square_ug_fitting( x, y, new_ug );

        return residual;
    }

#if 0
    //IDEA:
    //      after one fitting, merge similiar terms, i.e., modify ar to decrease the unknown variables, and try to fit again  -- only one time?
    //


    template< typename T = double >
    T two_layers_homotopy_forward_fitting(  matrix<std::size_t> const& ar,          // beam index matrix
                                            matrix<T> const& diag,                  // tilt matrix, row-major
                                            matrix<T> const& intensity,             // diffraction measurement, normalized, row-major
                                            matrix<T> const& ug,                    // initial guess
                                            std::complex<T> const& lhs_thickness,   // thickness for left hand side
                                            std::complex<T> const& rhs_thickness,   // thickness for right hand side
                                            matrix<T>& new_ug,                      // fitting result
                                            std::size_t column_index = 0 )          // column index in Scattering matrix
    {
        T current_residual    = backward_homotopy_details::make_two_layers_homotopy( ar, diag, intensity, ug, lhs_thickness, rhs_thickness, new_ug, column_index );
        matrix<T> test_ug{new_ug};
        T new_residual        = backward_homotopy_details::make_two_layers_homotopy( ar, diag, intensity, new_ug, lhs_thickness, rhs_thickness, new_ug, column_index );

        while( new_residual < current_residual )
        {
            current_residual    = new_residual;
            new_residual        = backward_homotopy_details::make_two_layers_homotopy( ar, diag, intensity, new_ug, lhs_thickness, rhs_thickness, new_ug, column_index );
            test_ug             = new_ug;
        }

        new_ug = test_ug;

        return current_residual;
    }

    template< typename T = double >
    T two_layers_homotopy_fitting(  matrix<std::size_t> const& ar, matrix<T> const& diag, matrix<T> const& intensity, matrix<T> const& ug, T const lhs_thickness, T const rhs_thickness, matrix<T>& new_ug, std::size_t column_index = 0 )
    {
        return two_layers_homotopy_fitting( ar, diag, intensity, ug, std::complex<T>{0, lhs_thickness}, std::complex<T>{0, rhs_thickness}, new_ug, column_index );
    }
#endif

}//namespace f

#endif//_BACKWARD_HOMOTOPY_HPP_INCLUDED_FADSOJASDKLJN3IU9HASDFKLJVAMBNVKJAHBSDFD

