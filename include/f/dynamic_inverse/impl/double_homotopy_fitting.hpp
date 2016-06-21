#ifndef MDOUBLE_HOMOTOPY_FITTING_HPP_INCLUDED_FDSPOIJASFDL498AFDLKJNSFDKLJVNAF9U 
#define MDOUBLE_HOMOTOPY_FITTING_HPP_INCLUDED_FDSPOIJASFDL498AFDLKJNSFDKLJVNAF9U

#include <f/matrix/matrix.hpp>
#include <f/dynamic_inverse/impl/double_square_modeling.hpp>
#include <f/dynamic_inverse/impl/extract_inner_product_coefficients.hpp>

#include <vector>
#include <complex>
#include <cstddef>
#include <vector>

namespace f
{
    template< typename T >
    struct double_homotopy_fitting
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

        double_homotopy_fitting( size_type const ug_size_ ) : ug_size( ug_size_ )
        {
            assert( ug_size );
            dsm.set_parameter_size( ug_size ); 
        }

        void register_entry(    size_matrix_type const& ar, 
                                value_type alpha, complex_matrix_type const& lhs_matrix, complex_matrix_type const& rhs_matrix, 
                                value_type beta, complex_matrix_type const& expm_matrix, 
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
            assert( alpha <= value_type{1} );
            assert( beta <= value_type{1} );
            assert( std::abs(alpha+beta-value_type{1}) < value_type{ 1.0e-10} );
            //assert( c1_matrix.row() == ar.row() );
            //assert( c1_matrix.col() == 1 );
            assert( expm_matrix.row() == ar.row() );
            assert( expm_matrix.col() == 1 );
            assert( column_index < ar.row() );

            size_type const n = ar.row();
            size_type const m = ug_size;

            matrix_type real_part(m, 1);
            matrix_type imag_part(m, 1);

            value_type norm_factor{0};
            //norm only one column
            //std::for_each( expm_matrix.col_begin( column_index ), expm_matrix.col_end( column_index ), [&norm_factor]( complex_type const& c ){ norm_factor += std::norm(c); } );
            std::for_each( expm_matrix.begin(), expm_matrix.end(), [&norm_factor]( complex_type const& c ){ norm_factor += std::norm(c); } );
            norm_factor /= static_cast<value_type>( expm_matrix.row() );

            for ( size_type r = 0; r != ar.row(); ++r )
            {
                //for \beta C/2 C/2 part
                extract_inner_product_coefficients( m, n, ar.row_begin(r), lhs_matrix.row_begin(r), rhs_matrix.col_begin(column_index), real_part.begin(), imag_part.begin() );
                real_part *= alpha;
                imag_part *= alpha;

                //for \gamma E part
                real_part[0][0] += beta * std::real( expm_matrix[r][column_index] );
                imag_part[0][0] += beta * std::imag( expm_matrix[r][column_index] );
                //real_part[0][0] += beta * std::real( expm_matrix[r][column_index] ) / norm_factor;
                //imag_part[0][0] += beta * std::imag( expm_matrix[r][column_index] ) / norm_factor;

                //needs modifying here
                dsm.register_entry( intensity[r][0], real_part.begin(), imag_part.begin() );
            }

#if 0
            //register lambda, ensuring lambda to be 1
            std::fill( real_part.begin(), real_part.end(), value_type{} );
            value_type const factor = value_type{1.0};
            value_type const weigh = factor * std::sqrt( static_cast<value_type>( intensity.row() ) );
            real_part[0][0] = weigh;
            imag_part[0][0] = weigh;
            dsm.register_entry( value_type{2} * weigh * weigh, real_part.begin(), imag_part.begin() );
#endif
        }

        void register_abs_entry( size_type const ug_index, value_type const ug_value, value_type progress_ratio )
        {
            size_type const m = ug_size;
            matrix_type real_part(m, 1);
            matrix_type imag_part(m, 1);

            std::fill( real_part.begin(), real_part.end(), value_type{} );
            std::fill( imag_part.begin(), imag_part.end(), value_type{} );

            value_type const weigh = ( value_type{1} - progress_ratio ) * std::sqrt( static_cast<value_type>(m) );
            value_type const intensity = value_type{2} * weigh * ug_value * weigh * ug_value;

            real_part[ug_index][0] = weigh;
            imag_part[ug_index][0] = weigh;

            dsm.register_entry( intensity, real_part.begin(), imag_part.begin() );
        }

        template< typename Input_Iterator >
        void set_initial_guess( Input_Iterator it_, Input_Iterator _it )
        {
            //dsm.guess( it_, _it );
            std::vector<value_type> arr{ it_, _it };
            arr[0] = value_type(1);
            std::for_each( arr.begin()+1, arr.end(), [](value_type& x) { while ( std::abs(x) > value_type{0.1}) { x /= value_type{10}; } } );
            dsm.guess( arr.begin(), arr.end() );
        }

        template< typename Output_Iterator >
        value_type output( Output_Iterator oi )
        {
            dsm.fit();
            dsm.output( oi );
            return dsm.residual();
        }
    
    };//struct double_homotopy_fitting

}//namespace f

#endif//_DOUBLE_HOMOTOPY_FITTING_HPP_INCLUDED_FDSPOIJASFDL498AFDLKJNSFDKLJVNAF9U

