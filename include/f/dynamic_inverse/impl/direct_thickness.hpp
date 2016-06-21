#ifndef UNDGKGLYLBVTJHIVTHVCEBVLOICBOOMPTPHGSTFSFFCEXGQAVJOPXOTESMYEMRSSVQRFIBESK
#define UNDGKGLYLBVTJHIVTHVCEBVLOICBOOMPTPHGSTFSFFCEXGQAVJOPXOTESMYEMRSSVQRFIBESK

//#include <f/device/assert/cuda_assert.hpp>

#include <f/pattern/pattern.hpp>
#include <f/coefficient/coefficient.hpp>
#include <f/coefficient/expm.hpp>
#include <f/dynamic_inverse/impl/structure_matrix.hpp>
#include <f/dynamic_inverse/impl/scattering_matrix.hpp>
#include <f/algorithm/for_each.hpp>

#include <functional>
#include <vector>
#include <cassert>
#include <thread>
#include <future>


//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <cuComplex.h>

namespace f
{
    
    template< typename T >
    struct direct_thickness
    {
        typedef T                                       value_type;
        typedef std::complex<T>                         complex_type;
        typedef value_type*                             pointer;
        typedef matrix<value_type>                      matrix_type;
        typedef matrix<complex_type>                    complex_matrix_type;
        typedef std::vector<matrix_type>                matrix_vector_type;
        typedef std::vector<complex_matrix_type>        complex_matrix_vector_type;
        typedef std::size_t                             size_type;
        typedef std::vector<size_type>                  size_vector_type;

        /*        
        typedef size_type*                              cu_size_pointer_type;
        typedef std::vector<cu_size_pointer_type>       cu_size_pointer_vector_type;   

        typedef cuDoubleComplex                         cu_complex_type;
        typedef cu_complex_type*                        cu_complex_pointer_type;
        typedef double                                  cu_value_type;
        typedef cu_value_type*                          cu_value_pointer_type;
        typedef std::vector<cu_complex_pointer_type>    cu_complex_pointer_vector_type;
        typedef std::vector<cu_value_pointer_type>      cu_value_pointer_vector_type;
        */

        pattern<value_type>const&               pt;
        value_type                              alpha;
        complex_type                            thickness;

        complex_matrix_type                     ug_cache;
        complex_matrix_vector_type              A_vec_cache;
        complex_matrix_vector_type              S_c1_vec_cache;     //[n][1]
        complex_matrix_vector_type              S_expm_vec_cache;   //[n][n]
        complex_matrix_vector_type              S_convex_vec_cache;
        matrix_vector_type                      I_homotopy_vec_cache;
        matrix_vector_type                      I_diff_vec_cache;

        void update_thickness( pointer p )
        {
            thickness = complex_type{ 0.0, p[pt.ug_size*2] };
        }

        std::function<value_type(pointer)> make_merit_function()
        {
            return [this]( pointer p )
            {
                (*this).update_thickness( p );
                (*this).update_ug( p );

                value_type residual{0};

                for( auto&& i_diff: (*this).I_diff_vec_cache )
                    residual += std::inner_product( i_diff.begin(), i_diff.end(), i_diff.begin(), value_type{0} );

                //TODO:homotopy

                return residual;
            };
        }

        std::function<value_type(pointer)> make_abs_function()
        {
            return [this]( pointer p )
            {
                (*this).update_thickness( p );
                (*this).update_ug( p );

                value_type residual{0};

                for( auto&& i_diff: (*this).I_diff_vec_cache )
                    residual += std::accumulate( i_diff.begin(), i_diff.end(), 0.0, []( value_type x, value_type y ){ return std::abs(x)+std::abs(y); } );

                //TODO:homotopy

                return residual;
            };
        }

        void config_alpha( value_type alpha_ )
        {
            alpha = alpha_;
        }

        direct_thickness( pattern<value_type> const& pt_, value_type alpha_ ) : pt( pt_ ), alpha( alpha_ ) 
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
                I_diff_vec_cache[index] = pt.intensity[index] - std::real( ug_cache[0][0] ) * I_homotopy_vec_cache[index] - std::imag(ug_cache[0][0]);
            }

        }

        void make_I_homotopy_vec_cache()
        {
            assert( I_homotopy_vec_cache.size() == pt.tilt_size );
            assert( S_expm_vec_cache.size() == pt.tilt_size );

            for ( size_type index = 0; index != pt.tilt_size; ++index )
            {
                I_homotopy_vec_cache[index].resize( S_expm_vec_cache[index].row(), 1 );
                std::transform( S_expm_vec_cache[index].begin(), S_expm_vec_cache[index].end(), I_homotopy_vec_cache[index].begin(), []( complex_type const& c) { return std::norm(c); } );
            }
        }

        void make_S_expm_vec_cache()
        {

            //unsigned long const threads = 3;
            //unsigned long const threads = 24;
            unsigned long const threads = 1;
            unsigned long const total_task = pt.tilt_size;
            unsigned long const task_per_thread = ( total_task + threads - 1 ) / threads;
            std::vector<std::thread> thread_array;

            auto fun = [this]( unsigned long starter, unsigned long ender )
            {
                for ( unsigned long index = starter; index != ender; ++index )
                    (*this).S_expm_vec_cache[index] = make_scattering_matrix( (*this).pt.ar[index], (*this).ug_cache, (*this).pt.diag[index].begin(), (*this).pt.diag[index].end(), (*this).thickness, (*this).pt.column_index );
                    //(*this).S_expm_vec_cache[index] = make_scattering_matrix( (*this).pt.ar[index], (*this).ug_cache, (*this).pt.diag[index].begin(), (*this).pt.diag[index].end(), (*this).pt.thickness, (*this).pt.column_index );

            };

            for ( unsigned long thread_index = 0; thread_index != threads; ++thread_index )
            {
                unsigned long const starter = task_per_thread * thread_index;
                unsigned long const ender = starter + task_per_thread > total_task ? total_task : starter + task_per_thread;
                if ( starter >= ender ) continue;
                thread_array.push_back( std::thread( fun, starter, ender ) );
            }

            for ( unsigned long thread_index = 0; thread_index != threads; ++thread_index )
                if ( thread_array[thread_index].joinable() )
                    thread_array[thread_index].join();

        }

        void make_A_vec_cache()
        {
            //
            //TODO: parallel here
            //
            for ( size_type index = 0; index != pt.tilt_size; ++index )
                A_vec_cache[index] = make_structure_matrix( pt.ar[index], ug_cache, pt.diag[index].begin(), pt.diag[index].end() );
        }

        void on_ug_changed()
        {
            make_A_vec_cache();
            make_S_expm_vec_cache();
            make_I_homotopy_vec_cache();
            make_I_diff_vec_cache();
        }

    };//struct direct_thickness

}//namespace f

#endif//UNDGKGLYLBVTJHIVTHVCEBVLOICBOOMPTPHGSTFSFFCEXGQAVJOPXOTESMYEMRSSVQRFIBESK

