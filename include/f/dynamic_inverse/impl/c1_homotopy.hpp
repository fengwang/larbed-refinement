#ifndef MC1_HOMOTOPY_HPP_INCLUDED_SDPOIASDLKJAS39OYHUASDKJASDFKLJHASDKLJAHDFKFDF
#define MC1_HOMOTOPY_HPP_INCLUDED_SDPOIASDLKJAS39OYHUASDKJASDFKLJHASDKLJAHDFKFDF

#include <f/pattern/pattern.hpp>
#include <f/coefficient/coefficient.hpp>
#include <f/coefficient/expm.hpp>
#include <f/dynamic_inverse/impl/structure_matrix.hpp>
#include <f/dynamic_inverse/impl/scattering_matrix.hpp>
#include <f/algorithm/for_each.hpp>

#include <functional>
#include <vector>
#include <cassert>

namespace f
{
    
    template< typename T >
    struct c1_homotopy
    {
        typedef T                                       value_type;
        typedef std::complex<T>                         complex_type;
        typedef value_type*                             pointer;
        typedef matrix<value_type>                      matrix_type;
        typedef matrix<complex_type>                    complex_matrix_type;
        typedef std::vector<matrix_type>                matrix_vector_type;
        typedef std::vector<complex_matrix_type>        complex_matrix_vector_type;
        typedef std::size_t                             size_type;

        pattern<value_type>const&               pt;
        value_type                              alpha;

        complex_matrix_type                     ug_cache;
        complex_matrix_vector_type              A_vec_cache;
        complex_matrix_vector_type              S_c1_vec_cache;     //[n][1]
        complex_matrix_vector_type              S_expm_vec_cache;   //[n][n]
        complex_matrix_vector_type              S_convex_vec_cache;
        matrix_vector_type                      I_homotopy_vec_cache;
        matrix_vector_type                      I_diff_vec_cache;

        std::function<value_type(pointer)> make_merit_function()
        {
            return [this]( pointer p )
            {
                (*this).update_ug( p );

                value_type residual{0};

                for( auto&& i_diff: (*this).I_diff_vec_cache )
                    residual += std::inner_product( i_diff.begin(), i_diff.end(), i_diff.begin(), value_type{0} );

                //fix??
                /*
                for ( auto&& ug : (*this).ug_cache )
                    residual += (*this).alpha * std::norm(ug);
                */

                return residual;
            };
            /*
            return [this]( pointer p )
            {
                (*this).update_ug( p );

                value_type residual{0};

                for( auto&& i_diff: (*this).I_diff_vec_cache )
                    residual += std::inner_product( i_diff.begin(), i_diff.end(), i_diff.begin(), value_type{0} );

                residual *= (*this).alpha;

                //fix??
                for ( auto&& ug : (*this).ug_cache )
                    residual += ( 1.0 - (*this).alpha ) * std::norm(ug);

                return residual;
            };
            */
        }

        void config_alpha( value_type alpha_ )
        {
            alpha = alpha_;
        }

        c1_homotopy( pattern<value_type> const& pt_, value_type alpha_ ) : pt( pt_ ), alpha( alpha_ ) 
        {
            ug_cache.resize( pt.ug_size, 1 );
            A_vec_cache.resize( pt.tilt_size );
            S_c1_vec_cache.resize( pt.tilt_size );
            S_expm_vec_cache.resize( pt.tilt_size );
            S_convex_vec_cache.resize( pt.tilt_size );
            I_homotopy_vec_cache.resize( pt.tilt_size );
            I_diff_vec_cache.resize( pt.tilt_size );
        }

        template< typename Itor >
        void update_ug( Itor begin )
        {
            pointer p = reinterpret_cast<pointer>( ug_cache.begin() );

            for ( size_type index = 0; index != pt.ug_size+pt.ug_size; ++index )
                *p++ = *begin++;

            on_ug_changed();
            //on_ug_changed_se();
        }

        void update_alpha( value_type alpha_ )
        {
            alpha = alpha_;
        }

        void make_I_diff_vec_cache()
        {
            for ( size_type index = 0; index != pt.tilt_size; ++index )
            {
                assert( pt.intensity[index].row() == I_homotopy_vec_cache[index].row() );
                assert( pt.intensity[index].col() == I_homotopy_vec_cache[index].col() );
                I_diff_vec_cache[index] = pt.intensity[index] - I_homotopy_vec_cache[index];
            }
        }

        void make_I_homotopy_vec_cache()
        {
            assert( I_homotopy_vec_cache.size() == pt.tilt_size );
            assert( S_convex_vec_cache.size() == pt.tilt_size );

            for ( size_type index = 0; index != pt.tilt_size; ++index )
            {
                I_homotopy_vec_cache[index].resize( S_convex_vec_cache[index].row(), 1 );
                std::transform( S_convex_vec_cache[index].begin(), S_convex_vec_cache[index].end(), I_homotopy_vec_cache[index].begin(), []( complex_type const& c) { return std::norm(c); } );
            }
        }

        void make_I_homotopy_vec_cache_se()
        {
            assert( I_homotopy_vec_cache.size() == pt.tilt_size );
            assert( S_expm_vec_cache.size() == pt.tilt_size );

            for ( size_type index = 0; index != pt.tilt_size; ++index )
            {
                I_homotopy_vec_cache[index].resize( S_expm_vec_cache[index].row(), 1 );
                std::transform( S_expm_vec_cache[index].begin(), S_expm_vec_cache[index].end(), I_homotopy_vec_cache[index].begin(), []( complex_type const& c) { return std::norm(c); } );
            }
        }

        void make_S_convex_vec_cache()
        {
            /*
            for ( size_type index = 0; index != pt.tilt_size; ++index )
            {
                S_convex_vec_cache[index].resize( S_c1_vec_cache[index].row(), 1 );
                for_each( S_convex_vec_cache[index].begin(), S_convex_vec_cache[index].end(),
                          S_c1_vec_cache[index].begin(), S_expm_vec_cache[index].col_begin( pt.column_index ), 
                          [this]( value_type& s, value_type s1, value_type s2) { s = ( value_type{1} - (*this).alpha ) * s1 + (*this).alpha * s2; } );
            }
            */
            assert( S_convex_vec_cache.size() == pt.tilt_size );
            complex_matrix_type cm;
            for ( size_type index = 0; index != pt.tilt_size; ++index )
            {
                cm.resize( S_c1_vec_cache[index].row(), 1 );
                for ( size_type r = 0; r != S_c1_vec_cache[index].row(); ++r )
                {
                    assert( S_c1_vec_cache[index].row() == S_expm_vec_cache[index].row() );
                    assert( S_c1_vec_cache[index].row() == S_expm_vec_cache[index].col() );
                    assert( S_c1_vec_cache[index].col() == 1 );
                    cm[r][0] = ( value_type{1} - alpha ) * S_c1_vec_cache[index][r][0] + alpha * S_expm_vec_cache[index][r][pt.column_index];
                }
                S_convex_vec_cache[index] = cm;
            }
        }

        void make_S_c1_vec_cache()
        {
            for ( size_type index = 0; index != pt.tilt_size; ++index )
                S_c1_vec_cache[index] = expm( A_vec_cache[index], pt.thickness, pt.column_index );
        }

        void make_S_expm_vec_cache()
        {
            for ( size_type index = 0; index != pt.tilt_size; ++index )
                S_expm_vec_cache[index] = make_scattering_matrix( pt.ar[index], ug_cache, pt.diag[index].begin(), pt.diag[index].end(), pt.thickness );
        }

        void make_A_vec_cache()
        {
            for ( size_type index = 0; index != pt.tilt_size; ++index )
                A_vec_cache[index] = make_structure_matrix( pt.ar[index], ug_cache, pt.diag[index].begin(), pt.diag[index].end() );
        }

        void on_ug_changed()
        {
            make_A_vec_cache();
            make_S_expm_vec_cache();
            make_S_c1_vec_cache();
            make_S_convex_vec_cache();
            make_I_homotopy_vec_cache();
            make_I_diff_vec_cache();
        }

        void on_ug_changed_se()
        {
            make_A_vec_cache();
            make_S_expm_vec_cache();
            make_I_homotopy_vec_cache_se();
            make_I_diff_vec_cache();
        }

    };//struct c1_homotopy

}//namespace f

#endif//_C1_HOMOTOPY_HPP_INCLUDED_SDPOIASDLKJAS39OYHUASDKJASDFKLJHASDKLJAHDFKFDF

