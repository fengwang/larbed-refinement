#ifndef MTRIPLE_HOMOTOPY_FITTING_HPP_INCLUDED_FDSPOIHASLKF4098YFASLKHASFDLKJH498 
#define MTRIPLE_HOMOTOPY_FITTING_HPP_INCLUDED_FDSPOIHASLKF4098YFASLKHASFDLKJH498

#include <f/matrix/matrix.hpp>
#include <f/dynamic_inverse/impl/double_square_modeling.hpp>
#include <f/dynamic_inverse/impl/extract_inner_product_coefficients.hpp>

#include <complex>
#include <cstddef>
#include <vector>

namespace f
{
    template< typename T >
    struct triple_homotopy_fitting
    {
        typedef T                               value_type;
        typedef std::complex<value_type>        complex_type;
        typedef std::size_t                     size_type;
        typedef matrix<size_type>               size_matrix_type;
        typedef matrix<complex_type>            complex_matrix_type;
        typedef matrix<value_type>              matrix_type;
        typedef std::vector<value_type>         vector_type;

        double_square_modeling<value_type>      dsm;
        size_type                               ug_size;

        triple_homotopy_fitting( size_type const ug_size_ ) : ug_size( ug_size_ )
        {
            assert( ug_size );
            dsm.set_parameter_size( ug_size ); 
        }

        void register_entry(    size_matrix_type const& ar, 
                                value_type alpha, complex_matrix_type const& c1_matrix, 
                                value_type beta, complex_matrix_type const& lhs_matrix, complex_matrix_type const& rhs_matrix, 
                                value_type gamma, complex_matrix_type const& expm_matrix, 
                                matrix_type const& intensity, size_type const column_index = 0 )
        {
            assert( ar.row() == ar.col() );
            assert( ar.row() == lhs_matrix.row() );
            assert( lhs_matrix.row() == lhs_matrix.col() );
            assert( ar.row() == rhs_matrix.row() );
            assert( ar.row() == intensity.row() );
            assert( 1 == intensity.col() );
            assert( (*(std::max_element(ar.begin(), ar.end()))) < ug_size );
            assert( alpha >= value_type{0} );
            assert( beta >= value_type{0} );
            assert( gamma >= value_type{0} );
            assert( alpha <= value_type{1} );
            assert( beta <= value_type{1} );
            assert( gamma <= value_type{1} );
            assert( std::abs(alpha+beta+gamma-value_type{1}) < value_type{ 1.0e-10} );
            assert( c1_matrix.row() == ar.row() );
            assert( c1_matrix.col() == 1 );
            assert( expm_matrix.row() == ar.row() );
            assert( expm_matrix.col() == 1 );
            assert( column_index < ar.row() );

            size_type const n = ar.row();
            size_type const m = ug_size;

            matrix_type real_part(m, 1);
            matrix_type imag_part(m, 1);

            for ( size_type r = 0; r != ar.row(); ++r )
            {
                //for \beta C/2 C/2 part
                extract_inner_product_coefficients( m, n, ar.row_begin(r), lhs_matrix.row_begin(r), rhs_matrix.col_begin(column_index), real_part.begin(), imag_part.begin() );
                real_part *= beta;
                imag_part *= beta;

                //for \alpha C1 part
                size_type const ug_index = ar[r][column_index];
                real_part[ug_index][0] += alpha * std::real( c1_matrix[r][0] );
                imag_part[ug_index][0] += alpha * std::imag( c1_matrix[r][0] );

                //for \gamma E part
                real_part[0][0] += gamma * std::real( expm_matrix[r][0] );
                imag_part[0][0] += gamma * std::imag( expm_matrix[r][0] );

                //needs modifying here
                dsm.register_entry( intensity[r][0], real_part.begin(), imag_part.begin() );
            }
        }


        template< typename Output_Iterator >
        value_type output( Output_Iterator oi )
        {
            dsm.fit();
            dsm.output( oi );
            return dsm.residual();
        }
    
    };//struct triple_homotopy_fitting


}//namespace f


#endif//_TRIPLE_HOMOTOPY_FITTING_HPP_INCLUDED_FDSPOIHASLKF4098YFASLKHASFDLKJH498

