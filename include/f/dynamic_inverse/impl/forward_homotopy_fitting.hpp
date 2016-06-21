#ifndef MFORWARD_HOMOTOPY_FITTING_HPP_INCLUDED_SPDOISAFLKJSDA43UHASFDKLJHSFDKLJHASDFKLJH
#define MFORWARD_HOMOTOPY_FITTING_HPP_INCLUDED_SPDOISAFLKJSDA43UHASFDKLJHSFDKLJHASDFKLJH

#include <f/matrix/matrix.hpp>
#include <f/dynamic_inverse/impl/double_square_modeling.hpp>
#include <f/dynamic_inverse/impl/double_square_fitting.hpp>
#include <f/dynamic_inverse/impl/extract_inner_product_coefficients.hpp>

#include <complex>
#include <cstddef>
#include <vector>

namespace f
{
    template< typename T >
    struct forward_homotopy_fitting
    {
        typedef T                               value_type;
        typedef std::complex<value_type>        complex_type;
        typedef std::size_t                     size_type;
        typedef matrix<size_type>               size_matrix_type;
        typedef matrix<complex_type>            complex_matrix_type;
        typedef matrix<value_type>              matrix_type;
        typedef std::vector<value_type>         vector_type;

        double_square_modeling<value_type>      dsm;
        //double_square_fitting<value_type>      dsm;
        size_type                               ug_size;

        forward_homotopy_fitting( size_type const ug_size_ ) : ug_size( ug_size_ )
        {
            assert( ug_size );
            dsm.set_parameter_size( ug_size ); 
        }

        void register_entry( size_matrix_type const& ar, complex_matrix_type const& lhs_matrix, complex_matrix_type const& rhs_matrix, matrix_type const& intensity, size_type const column_index = 0 )
        {
            assert( ar.row() == ar.col() );
            assert( ar.row() == lhs_matrix.row() );
            assert( lhs_matrix.row() == lhs_matrix.col() );
            assert( ar.row() == rhs_matrix.row() );
            //assert( 1 == rhs_matrix.col() );
            assert( ar.row() == intensity.row() );
            assert( 1 == intensity.col() );
            assert( (*(std::max_element(ar.begin(), ar.end()))) < ug_size );

            size_type const n = ar.row();
            size_type const m = ug_size;

            vector_type real_part(m);
            vector_type imag_part(m);

            for ( size_type r = 0; r != ar.row(); ++r )
            {
                //ommit the central column??
                if ( r == column_index ) continue;
                extract_inner_product_coefficients( m, n, ar.row_begin(r), lhs_matrix.row_begin(r), rhs_matrix.col_begin(column_index), real_part.begin(), imag_part.begin() );
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
    
    };//struct forward_homotopy_fitting


}//namespace f


#endif//_FORWARD_HOMOTOPY_FITTING_HPP_INCLUDED_SPDOISAFLKJSDA43UHASFDKLJHSFDKLJHASDFKLJH

