#ifndef QULHJAMAHYUVRGQTQKVYGHCKUUERVTCMMKNOVVYKEUNCMIJDINHJUNIDHBSUDQBVTYHOTHJNB
#define QULHJAMAHYUVRGQTQKVYGHCKUUERVTCMMKNOVVYKEUNCMIJDINHJUNIDHBSUDQBVTYHOTHJNB

#include <f/matrix/matrix.hpp>
#include <f/coefficient/coefficient_matrix.hpp>
#include <f/variate_generator/variate_generator.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cassert>
#include <set>

namespace f
{
    template< typename T >
    struct ug_initialization
    {
        typedef T                                       value_type;
        typedef std::complex<T>                         complex_type;
        typedef std::size_t                             size_type;
        typedef matrix<size_type>                       size_matrix_type;
        typedef matrix<value_type>                      matrix_type;

        size_type                                       ug_size;
        size_type                                       ar_dim;
        size_type                                       column_index;
        complex_type                                    thickness;
        matrix_type                                     diag_matrix;
        matrix_type                                     intensity_matrix;
        size_matrix_type                                ar;
        
        void check_all()
        {
            assert( ug_size );
            assert( ar_dim );
            assert( column_index < ar_dim );
            assert( std::abs( std::real(column_index) ) < value_type{1.0e-10} );
            assert( std::imag(column_index) > value_type{1.0e-10} );
            assert( diag_matrix.col() == ar_dim );
            assert( intensity_matrix.col() == ar_dim );
            assert( diag_matrix.row() == intensity_matrix.row() );
            assert( ar.row() == ar_dim );
            assert( ar.col() == ar_dim );
            assert( *std::max_element(ar.begin(), ar.end()) < ug_size );
        }

        template< typename Output_Iterator >
        void operator()( Output_Iterator oi )
        {
            matrix_type ug{ ug_size, 1, value_type{} };

            //random generated value
            //variate_generator<value_type> vg{ value_type{-0.01}, value_type{0.01} };
            //std::generate( ug.begin(), ug.end(), vg );

            //factor
            value_type const total_intensity = std::accumulate( intensity_matrix.begin(), intensity_matrix.end(), value_type{} );
            ug[0][0] = value_type{static_cast<value_type>(intensity_matrix.row())} / total_intensity;

            for ( size_type r = 0; r != intensity_matrix.row(); ++r )
                for ( size_type c = 0; c != intensity_matrix.col(); ++c )
                {
                    //size_type const ug_index = ar[r][c];
                    size_type const ug_index = ar[c][column_index];
                    if ( ! ug_index ) continue;
                    complex_type const& factor = make_coefficient_element( thickness, diag_matrix.row_begin(r), diag_matrix.row_end(r), c, column_index );
                    value_type ug_rc = std::sqrt( intensity_matrix[r][c] / std::norm( factor ) );
                    ug[ug_index][0] += ug_rc * intensity_matrix[r][c];
                }

            for ( size_type index = 0; index != ar_dim; ++index )
            {
                size_type const ug_index = ar[index][column_index];
                if ( ! ug_index ) continue;
                value_type const coef = std::accumulate( intensity_matrix.col_begin(index), intensity_matrix.col_end(index), value_type{} );
                ug[ug_index][0] /= coef;
            }

            std::copy( ug.begin(), ug.end(), oi );
        }

        void config_ar( size_matrix_type const& ar_ )
        {
            ar = ar_;
            //update ug size
            std::set<size_type> ug_set;
            for ( auto&& index : ar )
                ug_set.insert( index );

            ug_size = ug_set.size();
        }

        void config_intensity_matrix( matrix_type const& intensity_matrix_ )
        {
            intensity_matrix = intensity_matrix_;
        }

        void config_diag_matrix( matrix_type const& diag_matrix_ )
        {
            diag_matrix = diag_matrix_;
        }

        void config_thickness( complex_type const& thickness_ )
        {
            thickness = thickness_;
        }

        void config_column_index( size_type const column_index_ )
        {
            column_index = column_index_;
        }

        void config_ar_dim( size_type const ar_dim_ )
        {
            ar_dim = ar_dim_;
        }

        void config_ug_size( size_type const ug_size_ )
        {
            ug_size = ug_size_;
        }

    };//struct ug_initialization

}//namespace f

#endif//QULHJAMAHYUVRGQTQKVYGHCKUUERVTCMMKNOVVYKEUNCMIJDINHJUNIDHBSUDQBVTYHOTHJNB

