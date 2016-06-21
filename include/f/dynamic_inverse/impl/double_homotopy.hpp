#ifndef MDOUBLE_HOMOTOPY_HPP_INCLUDED_SDPOI3498YAFKLJSFDKMVCNIOUAHFD98Y43SAILUFH
#define MDOUBLE_HOMOTOPY_HPP_INCLUDED_SDPOI3498YAFKLJSFDKMVCNIOUAHFD98Y43SAILUFH

#include <f/dynamic_inverse/impl/double_homotopy_fitting.hpp>
#include <f/dynamic_inverse/impl/structure_matrix.hpp>
#include <f/dynamic_inverse/impl/scattering_matrix.hpp>
#include <f/dynamic_inverse/impl/ug_initialization.hpp>
#include <f/algorithm/for_each.hpp>
#include <f/coefficient/coefficient_matrix.hpp>
#include <f/coefficient/expm.hpp>
#include <f/matrix/matrix.hpp>

#include <functional>
#include <complex>
#include <cstddef>
#include <vector>
#include <map>

namespace f
{
    /*
       the model is 
            S = \alpha(x) S_{1/2} S_{1/2} + \beta(x) E
    */

    template< typename T >
    struct double_homotopy
    {
        typedef T                                               value_type;
        typedef std::size_t                                     size_type;
        typedef std::complex<value_type>                        complex_type;
        typedef matrix<value_type>                              matrix_type;
        typedef matrix<size_type>                               size_matrix_type;
        typedef matrix<complex_type>                            complex_matrix_type;
        typedef std::vector<matrix_type>                        matrix_vector_type;
        typedef std::vector<value_type>                         vector_type;
        typedef std::function<value_type(value_type)>           function_type;
        typedef std::map<size_type, value_type>                 ug_c1_approximation_type;

        size_type                           ug_size;
        size_type                           ar_dim;
        size_type                           column_index;
        complex_type                        thickness;
        matrix_type                         diag_matrix;        //row order
        matrix_type                         intensity_matrix;   //row order
        size_matrix_type                    ar;
        matrix_type                         initial_ug;         //column matrix
        matrix_type                         new_ug;
        value_type                          new_residual;

        value_type                          progress_ratio;     //current stage in the fitting progress, from 0 to 1
        size_type                           max_iteration;

        // \alpha(x) + \beta(x) = 1
        function_type                       alpha;
        function_type                       beta;
        function_type                       gamma;

        ug_c1_approximation_type            ug_c1_approximation;

        void config_progress_ratio( value_type progress_ratio_ )
        {
            progress_ratio = progress_ratio_;
        }

        template < typename Output_Iterator >
        value_type output( Output_Iterator oit )
        {
            generate_c1_guess();
            fit();
            std::copy( new_ug.begin(), new_ug.end(), oit );
            return new_residual;
        }

        void generate_c1_guess()
        {
            ug_initialization<value_type> ui;
            ui.config_ar( ar );
            ui.config_intensity_matrix( intensity_matrix );
            ui.config_diag_matrix( diag_matrix );
            ui.config_thickness( thickness );
            ui.config_column_index( column_index );
            ui.config_ar_dim( ar_dim );
            ui.config_ug_size( ug_size );
            matrix_type new_ug( ug_size, 1 );
            ui( new_ug.begin() );
            for ( size_type index = 0; index != ar_dim; ++index )
            {
                size_type const ug_index = ar[index][column_index];
                if ( 0 == ug_index ) continue;
                ug_c1_approximation[ug_index] = new_ug[ug_index][0];
            }
        }

        void fit()
        {
            assert( ug_size );
            assert( ar_dim );
            assert( column_index < ar_dim );
            assert( std::abs(std::real(thickness)) < 1.0e-10 );
            assert( std::imag(thickness) > 1.0e-10 );
            assert( diag_matrix.col() == ar_dim );
            assert( diag_matrix.row() == intensity_matrix.row() );
            assert( intensity_matrix.col() == ar_dim );
            assert( initial_ug.row() == ug_size );
            assert( initial_ug.col() == 1 );
            assert( ar.row() == ar.col() );
            assert( ar_dim == ar.row() );
            assert( progress_ratio >= value_type{0} );
            assert( progress_ratio <= value_type{1} );
            assert( alpha );
            assert( beta );

            new_residual = iterate( initial_ug, new_ug );
        }

        value_type iterate( matrix_type const& initial_matrix, matrix_type& result_matrix )
        {
            double_homotopy_fitting<value_type> dhf{ug_size};

            size_type const tilt_number = diag_matrix.row();

            matrix_type intensity{ intensity_matrix.col(), 1 };

            for ( size_type index = 0; index != tilt_number; ++index )
            {
                std::copy( intensity_matrix.row_begin(index), intensity_matrix.row_end(index), intensity.col_begin(0) );

                //TODO -- optimizaton here
                dhf.register_entry( ar, 
                                    //C/2 * C/2 approximation
                                    alpha(progress_ratio), make_coefficient_matrix( thickness/2.0, diag_matrix.row_begin(index), diag_matrix.row_end(index) ), expm( make_structure_matrix(ar, initial_matrix, diag_matrix.row_begin(index), diag_matrix.row_end(index) ), thickness/2.0, column_index ),
                                    //standard expm
                                    beta(progress_ratio), make_scattering_matrix( ar, initial_matrix, diag_matrix.row_begin(index), diag_matrix.row_end(index), thickness, column_index ),
                                    intensity, column_index );
            }

#if 0
            for ( auto const& elem : ug_c1_approximation )
            {
                size_type const ug_index = elem.first;
                value_type const ug_value = elem.second;
                dhf.register_abs_entry( ug_index, ug_value, progress_ratio );
            }
#endif

            dhf.set_initial_guess( initial_matrix.begin(), initial_matrix.end() );

            result_matrix.resize( ug_size, 1 );
            value_type const residual = dhf.output( result_matrix.begin() );

#if 0
            std::cout << "\n current residual is " << residual << "\n"; 
            std::cout << "\n current ug is \n" << result_matrix.transpose() << "\n"; 
#endif

            return residual;
        }

        void config_ug_size( size_type const ug_size_)
        {
            ug_size = ug_size_;
        }

        void config_ar_dim( size_type const ar_dim_)
        {
            ar_dim = ar_dim_;
        }

        void config_column_index( size_type const column_index_)
        {
            column_index = column_index_;
        }

        void config_thickness( complex_type const& thickness_ )
        {
            thickness = thickness_;
        }

        void config_diag_matrix( matrix_type const& diag_matrix_ )
        {
            diag_matrix = diag_matrix_;
        }

        void config_intensity_matrix( matrix_type const intensity_matrix_ )
        {
            intensity_matrix = intensity_matrix_;
        }

        void config_ar( size_matrix_type const& ar_ )
        {
            ar = ar_;
        }

        void config_initial_ug( matrix_type const& initial_ug_ )
        {
            initial_ug = initial_ug_;
        }

        void config_max_iteration( size_type const max_iteration_ )
        {
            max_iteration = max_iteration_;
        }

        template< typename Func >
        void config_gamma( Func gamma_ )
        {
            gamma = gamma_;
        }

        template< typename Func >
        void config_beta( Func beta_ )
        {
            beta = beta_;
        }

        template< typename Func >
        void config_alpha( Func alpha_ )
        {
            alpha = alpha_;
        }
    
    };//struct double_homotopy

}//namespace f

#endif//_DOUBLE_HOMOTOPY_HPP_INCLUDED_SDPOI3498YAFKLJSFDKMVCNIOUAHFD98Y43SAILUFH

