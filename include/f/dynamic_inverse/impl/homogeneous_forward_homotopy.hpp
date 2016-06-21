#ifndef QPXEPFMCNBCTRMEVOANHAPNPQATJOINOETJJOETPSQVYNCWKMVQVKILBLDUGPHHXILXSYCXDN
#define QPXEPFMCNBCTRMEVOANHAPNPQATJOINOETJJOETPSQVYNCWKMVQVKILBLDUGPHHXILXSYCXDN

#include <f/dynamic_inverse/impl/forward_homotopy_fitting.hpp>
#include <f/dynamic_inverse/impl/structure_matrix.hpp>
#include <f/algorithm/for_each.hpp>
#include <f/coefficient/coefficient_matrix.hpp>
#include <f/matrix/matrix.hpp>

#include <complex>
#include <cstddef>
#include <vector>

namespace f
{
    template< typename T >
    struct homogeneous_forward_homotopy
    {
        typedef T                           value_type;
        typedef std::size_t                 size_type;
        typedef std::complex<value_type>    complex_type;
        typedef matrix<value_type>          matrix_type;
        typedef matrix<size_type>           size_matrix_type;
        typedef matrix<complex_type>        complex_matrix_type;
        typedef std::vector<matrix_type>    matrix_vector_type;
        typedef std::vector<value_type>     vector_type;

        size_type                           ug_size;
        size_type                           ar_dim;
        size_type                           column_index;
        complex_type                        lhs_thickness;
        complex_type                        rhs_thickness;
        matrix_type                         diag_matrix;        //row order
        matrix_type                         intensity_matrix;   //row order
        size_matrix_type                    ar;
        matrix_type                         initial_ug;         //column matrix
        matrix_type                         new_ug;
        value_type                          new_residual;

        size_type                           max_iteration;

        homogeneous_forward_homotopy(): max_iteration( 5 ) {}

        template < typename Output_Iterator >
        value_type output( Output_Iterator oit )
        {
            fit();
            std::copy( new_ug.begin(), new_ug.end(), oit );
            return new_residual;
        }

        void fit()
        {
            assert( ug_size );
            assert( ar_dim );
            assert( column_index < ar_dim );
            assert( std::abs(std::real(lhs_thickness)) < 1.0e-10 );
            assert( std::abs(std::real(rhs_thickness)) < 1.0e-10 );
            assert( std::imag(lhs_thickness) > 1.0e-10 );
            assert( std::imag(rhs_thickness) > 1.0e-10 );
            assert( diag_matrix.col() == ar_dim );
            assert( diag_matrix.row() == intensity_matrix.row() );
            assert( intensity_matrix.col() == ar_dim );
            assert( initial_ug.row() == ug_size );
            assert( initial_ug.col() == 1 );
            assert( ar.row() == ar.col() );
            assert( ar_dim == ar.row() );

            new_residual = iterate( initial_ug, new_ug );

            matrix_type second_ug{ initial_ug };

            size_type current_iteration = 0;

            matrix_vector_type  vm;
            vector_type         vr;

            vm.push_back( new_ug );
            vr.push_back( new_residual );

            value_type best_residual_so_far = new_residual;

            while ( true )
            {
                value_type const second_residual = iterate( new_ug, second_ug );

                bool break_flag = false;

                //??
                if ( best_residual_so_far > max_iteration * second_residual ) break_flag = true;

                best_residual_so_far = std::min( second_residual, best_residual_so_far );

                if( ++current_iteration  > max_iteration ) break_flag = true;

                new_ug.swap( second_ug );
                new_residual = second_residual;

                vm.push_back( new_ug );
                vr.push_back( new_residual );

                if ( break_flag ) break;
            }

            size_type const elite_index = std::distance( vr.begin(), std::min_element( vr.begin(), vr.end() ) );
            std::copy( vm[elite_index].begin(), vm[elite_index].end(), new_ug.begin() );
            
            std::cout << "\ncurrent elite residual is " << vr[elite_index] << ", at iteration " << current_iteration <<  std::endl;
        }

        value_type iterate( matrix_type const& initial_matrix, matrix_type& result_matrix )
        {
            forward_homotopy_fitting<value_type> fhf{ug_size};

            size_type const tilt_number = diag_matrix.row();

            matrix_type intensity{ intensity_matrix.col(), 1 };

            for ( size_type index = 0; index != tilt_number; ++index )
            {
                complex_matrix_type const& lhs_coefficient_matrix = make_coefficient_matrix( lhs_thickness, diag_matrix.row_begin(index), diag_matrix.row_end(index) );
                complex_matrix_type const& rhs_scattering_matrix = make_scattering_matrix( ar, initial_matrix, diag_matrix.row_begin(index), diag_matrix.row_end(index), rhs_thickness );
                std::copy( intensity_matrix.row_begin(index), intensity_matrix.row_end(index), intensity.col_begin(0) );

                fhf.register_entry( ar, lhs_coefficient_matrix, rhs_scattering_matrix, intensity, column_index );
            }

            result_matrix.resize( ug_size, 1 );
            value_type const residual = fhf.output( result_matrix.begin() );

            std::cout << "\n current residual is " << residual << "\n"; 
            std::cout << "\n current ug is \n" << result_matrix.transpose() << "\n"; 

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

        void config_lhs_thickness( complex_type const& lhs_thickness_ )
        {
            lhs_thickness = lhs_thickness_;
        }

        void config_rhs_thickness( complex_type const& rhs_thickness_ )
        {
            rhs_thickness = rhs_thickness_;
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
    
    };//struct homogeneous_forward_homotopy

}//namespace f

#endif//QPXEPFMCNBCTRMEVOANHAPNPQATJOINOETJJOETPSQVYNCWKMVQVKILBLDUGPHHXILXSYCXDN

