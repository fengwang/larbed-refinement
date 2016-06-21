#ifndef JAVMKUPNVXKHKEQRPRFWPJYOCVNHCLMMOAFTBVFQKMSIIGLQSNGJVRXVLVGQTSIFNXOCVKGYX
#define JAVMKUPNVXKHKEQRPRFWPJYOCVNHCLMMOAFTBVFQKMSIIGLQSNGJVRXVLVGQTSIFNXOCVKGYX

#include <f/device/assert/cuda_assert.hpp>

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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

namespace f
{
    
    template< typename T >
    struct direct_sd
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
        
        typedef size_type*                              cu_size_pointer_type;
        typedef std::vector<cu_size_pointer_type>       cu_size_pointer_vector_type;   

        typedef cuDoubleComplex                         cu_complex_type;
        typedef cu_complex_type*                        cu_complex_pointer_type;
        typedef double                                  cu_value_type;
        typedef cu_value_type*                          cu_value_pointer_type;
        typedef std::vector<cu_complex_pointer_type>    cu_complex_pointer_vector_type;
        typedef std::vector<cu_value_pointer_type>      cu_value_pointer_vector_type;

        pattern<value_type>                     pt;
        //pattern<value_type>const&               pt;
        value_type                              alpha;

        complex_matrix_type                     ug_cache;
        complex_matrix_vector_type              A_vec_cache;
        complex_matrix_vector_type              S_c1_vec_cache;     //[n][1]
        complex_matrix_vector_type              S_expm_vec_cache;   //[n][n]
        complex_matrix_vector_type              S_convex_vec_cache;
        matrix_vector_type                      I_homotopy_vec_cache;
        matrix_vector_type                      I_diff_vec_cache;

#if 0

        cu_value_pointer_type                   residual;

        //cuda space
        size_type                               align_dim;
        cu_size_pointer_type                    cuda_dim_cache;
        cu_complex_pointer_type                 cuda_ug_cache;
        cu_complex_pointer_type                 cuda_S_cache;
        cu_complex_pointer_type                 cuda_diag_cache;
        cu_value_pointer_type                   cuda_I_new_cache;
        cu_value_pointer_type                   cuda_I_ori_cache;
        cu_size_pointer_type                    cuda_ar_cache;

        cu_complex_pointer_type                 cuda_A_cache;
        cu_complex_pointer_type                 cuda_a_cache;       //A after scaling
        cu_complex_pointer_type                 cuda_aa_cache;
        cu_complex_pointer_type                 cuda_aaa_cache;
        cu_complex_pointer_type                 cuda_PP_cache;      //polynomials
        cu_complex_pointer_type                 cuda_PQ_cache;
        cu_complex_pointer_type                 cuda_PR_cache;
        cu_complex_pointer_type                 cuda_PS_cache;

        void on_ug_changed_in_gpu()
        {
            //copy new_ug to cuda_ug_cache
            ug_cache[0] = complex_type{0.0, 0.0};
            cuda_assert( cudaMemcpy( cuda_ug_cache, ug_cache.data(), pt.ug_size, cudaMemcpyHostToDevice ) );

            //extern "C" void direct_sd_update();



            //
            //1) update A
            //2) update S
            //3) update I_new
            //4) calc residual
        }

        void make_gpu_memory_cache()
        {
            cuda_assert( cudaMallocManaged( (void**)residual, sizeof(cu_value_type) ) );

            for ( size_type tilt_index = 0; tilt_index != pt.tilt_size; ++tilt_index )
                align_dim = std::max( align_dim, pt.ar[tilt_index].row() );

            align_dim = ( align_dim & 0x11111 ) ? ( ( ( align_dim >> 5 ) + 1 ) << 5 ) : align_dim;

            size_type const n_dim = sizeof( size_type ) * pt.tilt_size;
            cuda_assert( cudaMalloc( (void**)cuda_dim_cache, n_dim ) );
            cuda_assert( cudaMemSet( cuda_dim_cache, 0, n_dim ) );

            size_type const n_ug = sizeof( cu_complex_type ) * pt.ug_size;
            cuda_assert( cudaMalloc( (void**)cuda_ug_cache, n_ug ) );
            cuda_assert( cudaMemSet( cuda_ug_cache, 0, n_ug ) );

            //cache for expm
            size_type const n_S = sizeof( cu_complex_type ) * align_dim * align_dim * pt.tilt_size;
            cuda_assert( cudaMalloc( (void**)cuda_A_cache, n_S ) );
            cuda_assert( cudaMemSet( cuda_A_cache, 0, n_S ) );
            cuda_assert( cudaMalloc( (void**)cuda_a_cache, n_S ) );
            cuda_assert( cudaMemSet( cuda_a_cache, 0, n_S ) );
            cuda_assert( cudaMalloc( (void**)cuda_aa_cache, n_S ) );
            cuda_assert( cudaMemSet( cuda_aa_cache, 0, n_S ) );
            cuda_assert( cudaMalloc( (void**)cuda_aaa_cache, n_S ) );
            cuda_assert( cudaMemSet( cuda_aaa_cache, 0, n_S ) );
            cuda_assert( cudaMalloc( (void**)cuda_PP_cache, n_S ) );
            cuda_assert( cudaMemSet( cuda_PP_cache, 0, n_S ) );
            cuda_assert( cudaMalloc( (void**)cuda_PQ_cache, n_S ) );
            cuda_assert( cudaMemSet( cuda_PQ_cache, 0, n_S ) );
            cuda_assert( cudaMalloc( (void**)cuda_PR_cache, n_S ) );
            cuda_assert( cudaMemSet( cuda_PR_cache, 0, n_S ) );
            cuda_assert( cudaMalloc( (void**)cuda_PS_cache, n_S ) );
            cuda_assert( cudaMemSet( cuda_PS_cache, 0, n_S ) );
            cuda_assert( cudaMalloc( (void**)cuda_S_cache, n_S ) );
            cuda_assert( cudaMemSet( cuda_S_cache, 0, n_S ) );
            //

            size_type const n_diag = sizeof( cu_complex_type ) * align_dim * pt.tilt_size;
            cuda_assert( cudaMalloc( (void**)cuda_diag_cache, n_diag ) );
            cuda_assert( cudaMemSet( cuda_diag_cache, 0, n_diag ) );

            size_type const n_I_new = sizeof( cu_value_type ) * align_dim * pt.tilt_size;
            cuda_assert( cudaMalloc( (void**)cuda_I_new_cache, n_I_new ) );
            cuda_assert( cudaMemSet( cuda_I_new_cache, 0, n_I_new ) );

            size_type const n_I_ori = sizeof( cu_value_type ) * align_dim * pt.tilt_size;
            cuda_assert( cudaMalloc( (void**)cuda_I_ori_cache, n_I_ori ) );
            cuda_assert( cudaMemSet( cuda_I_ori_cache, 0, n_I_ori ) );

            size_type const n_ar = sizeof( size_type ) * align_dim * align_dim * pt.tilt_size;
            cuda_assert( cudaMalloc( (void**)cuda_ar_cache, n_ar ) );
            cuda_assert( cudaMemSet( cuda_ar_cache, 0, n_I_ar ) );

            //clean all the gpu memory here

            size_vector_type dim_cache;

            for ( size_type tilt_index = 0; tilt_index != pt.tilt_size; ++tilt_index )
            {
                size_type const dim_ar = pt.ar[tilt_index].row();
                dim_cache.push_back( dim_ar );

                //Intensity -- constant
                cu_value_pointer_type p_I_ori = cuda_I_ori_cache + align_dim * tilt_index;
                cuda_assert( cudaMemcpy( p_I_ori, pt.intensity[tilt_index].data(), n_I_ori, cudaMemcpyHostToDevice ) );

                //Ar -- constant
                cu_size_pointer_type p_ar = cuda_ar_cache + align_dim * align_dim * tilt_index;
                cuda_assert( cudaMemcpy( p_ar, pt.ar[tilt_index].data(), n_ar, cudaMemcpyHostToDevice ) );

                //Diag -- constant
                cu_complex_pointer_type p_dia = cuda_diag_cache + align_dim * tilt_index;
                cuda_assert( cudaMemcpy( p_dia, pt.diag[tilt_index].data(), n_diag, cudaMemcpyHostToDevice ) );
            }

            cuda_assert( cudaMemcpy( cuda_diag_cache, dim_cache.data(), n_dim, cudaMemcpyHostToDevice ) );
        }

        void clean_gpu_memory_cache()
        {
            cuda_assert( cudaFree( residual ) );
            cuda_assert( cudaFree( cuda_ug_cache ) );
            cuda_assert( cudaFree( cuda_S_cache ) );
            cuda_assert( cudaFree( cuda_diag_cache ) );
            cuda_assert( cudaFree( cuda_I_new_cache ) );
            cuda_assert( cudaFree( cuda_I_ori_cache ) );
            cuda_assert( cudaFree( cuda_ar_cache ) );
            cuda_assert( cudaFree( cuda_dim_cache ) );
            cuda_assert( cudaFree( cuda_A_cache ) );
            cuda_assert( cudaFree( cuda_a_cache ) );
            cuda_assert( cudaFree( cuda_aa_cache ) );
            cuda_assert( cudaFree( cuda_aaa_cache ) );
            cuda_assert( cudaFree( cuda_PP_cache ) );
            cuda_assert( cudaFree( cuda_PQ_cache ) );
            cuda_assert( cudaFree( cuda_PR_cache ) );
            cuda_assert( cudaFree( cuda_PS_cache ) );
        }

        ~direct_sd()
        {
            clean_gpu_memory_cache();
        }

        std::function<value_type(pointer)> make_cuda_merit_function()
        {
            if ( ! align_dim ) //align_dim is zero if cuda memory not set
                make_gpu_memory_cache();

            return [this]( pointer p )
            {
                /*
                make_A_vec_cache();
                make_S_expm_vec_cache();
                make_I_homotopy_vec_cache();
                make_I_diff_vec_cache();
                reduce_residual();
                */

                return residual;
            };
        }
#endif

        void update_thickness( value_type val )
        {
            pt.update_thickness( val );
        }

        std::function<value_type(pointer)> make_merit_function()
        {
            return [this]( pointer p )
            {
                (*this).update_thickness( *(p+(*this).pt.ug_size*2) );
                (*this).update_ug( p );

                value_type residual{0};

                for( auto&& i_diff: (*this).I_diff_vec_cache )
                    //new abs residual
                    //residual += std::accumulate( i_diff.begin(), i_diff.end(), value_type(0), [](value_type x, value_type y){ return std::abs(x)+std::abs(y); } );
                    //old squared residual
                     residual += std::inner_product( i_diff.begin(), i_diff.end(), i_diff.begin(), value_type{0} );

                //TODO:homotopy

                return residual;
            };
        }

        void config_alpha( value_type alpha_ )
        {
            alpha = alpha_;
        }

        direct_sd( pattern<value_type> const& pt_, value_type alpha_ ) : pt( pt_ ), alpha( alpha_ ) 
        {
            ug_cache.resize( pt.ug_size, 1 );
            A_vec_cache.resize( pt.tilt_size );
            S_c1_vec_cache.resize( pt.tilt_size );
            S_expm_vec_cache.resize( pt.tilt_size );
            S_convex_vec_cache.resize( pt.tilt_size );
            I_homotopy_vec_cache.resize( pt.tilt_size );
            I_diff_vec_cache.resize( pt.tilt_size );

            //align_dim = 0;
        }

        template< typename Itor >
        void update_ug( Itor begin )
        {
            pointer p = reinterpret_cast<pointer>( ug_cache.begin() );

            for ( size_type index = 0; index != pt.ug_size+pt.ug_size; ++index )
                *p++ = *begin++;

            //std::for_each( ug_cache.begin(), ug_cache.end(), []( complex_type& c ) { while( std::norm(c) > 0.1 ) c /= 10.0; });

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
                //I_diff_vec_cache[index] = pt.intensity[index] - I_homotopy_vec_cache[index];
                I_diff_vec_cache[index] = pt.intensity[index] - std::real( ug_cache[0][0] ) * I_homotopy_vec_cache[index] - std::imag(ug_cache[0][0]);
            }

            //std::cerr << "\nmake_I_diff_vec_cache finished.\n";
        }

/*
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
*/

        void make_I_homotopy_vec_cache()
        {
            assert( I_homotopy_vec_cache.size() == pt.tilt_size );
            assert( S_expm_vec_cache.size() == pt.tilt_size );

            for ( size_type index = 0; index != pt.tilt_size; ++index )
            {
                I_homotopy_vec_cache[index].resize( S_expm_vec_cache[index].row(), 1 );
                //std::transform( S_expm_vec_cache[index].col_begin(pt.column_index), S_expm_vec_cache[index].col_end(pt.column_index), I_homotopy_vec_cache[index].col_begin(0), []( complex_type const& c) { return std::norm(c); } );
                std::transform( S_expm_vec_cache[index].begin(), S_expm_vec_cache[index].end(), I_homotopy_vec_cache[index].begin(), []( complex_type const& c) { return std::norm(c); } );
            }

            //std::cerr << "\nmake_I_homotopy_vec_cache finished.\n";
        }

/*
        void make_S_convex_vec_cache()
        {
#if 0
            for ( size_type index = 0; index != pt.tilt_size; ++index )
            {
                S_convex_vec_cache[index].resize( S_c1_vec_cache[index].row(), 1 );
                for_each( S_convex_vec_cache[index].begin(), S_convex_vec_cache[index].end(),
                          S_c1_vec_cache[index].begin(), S_expm_vec_cache[index].col_begin( pt.column_index ), 
                          [this]( value_type& s, value_type s1, value_type s2) { s = ( value_type{1} - (*this).alpha ) * s1 + (*this).alpha * s2; } );
            }
#endif
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
*/

        void make_S_expm_vec_cache()
        {
            //std::cerr << "\nexecuting make_S_expm_vec_cache ........\n";

#if  0      //not work with mac/g++           
            std::vector<std::future<complex_matrix_type>> cm_fut_arr;

            for ( size_type index = 0; index != pt.tilt_size; ++index )
                cm_fut_arr.emplace_back( std::async( std::launch::async, 
                                                     //[]( auto const& ar, auto const& ug_cache, auto diag_begin, auto diag_end, auto thickness, auto column_index )
                                                     []( decltype(pt.ar[index]) ar, decltype(ug_cache) ug_cache, decltype( pt.diag[index].begin() ) diag_begin, decltype( pt.diag[index].end()) diag_end, 
                                                         decltype(pt.thickness) thickness, decltype( pt.column_index  ) column_index )
                                                     {
                                                        return make_scattering_matrix( ar, ug_cache, diag_begin, diag_end, thickness, column_index );
                                                     },
                                                     pt.ar[index], ug_cache, pt.diag[index].begin(), pt.diag[index].end(), pt.thickness, pt.column_index
                                                    )
                                       );
            for ( size_type index = 0; index != pt.tilt_size; ++index )
                S_expm_vec_cache[index] = cm_fut_arr[index].get();


#endif

#if 1       //std::threads
            unsigned long const threads = 24;
            unsigned long const total_task = pt.tilt_size;
            unsigned long const task_per_thread = ( total_task + threads - 1 ) / threads;
            std::vector<std::thread> thread_array;

            auto fun = [this]( unsigned long starter, unsigned long ender )
            {
                for ( unsigned long index = starter; index != ender; ++index )
                    (*this).S_expm_vec_cache[index] = make_scattering_matrix( (*this).pt.ar[index], (*this).ug_cache, (*this).pt.diag[index].begin(), (*this).pt.diag[index].end(), (*this).pt.thickness, (*this).pt.column_index );

            };

            for ( unsigned long thread_index = 0; thread_index != threads; ++thread_index )
            {
                unsigned long const starter = task_per_thread * thread_index;
                unsigned long const ender = starter + task_per_thread > total_task ? total_task : starter + task_per_thread;
                thread_array.push_back( std::thread( fun, starter, ender ) );
            }
            for ( unsigned long thread_index = 0; thread_index != threads; ++thread_index )
                thread_array[thread_index].join();
#endif

#if 0 
            //
            //TODO: parallel here
            //
            for ( size_type index = 0; index != pt.tilt_size; ++index )
            {
                //std::cerr << "making index " << index << "\n";

                //
                S_expm_vec_cache[index] = make_scattering_matrix( pt.ar[index], ug_cache, pt.diag[index].begin(), pt.diag[index].end(), pt.thickness, pt.column_index );
                //


                //std::cerr << "index " << index << " matrix done.\n";
            }

            //std::cerr << "\nmake_S_expm_vec_cache finished.\n";
#endif            
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

    };//struct direct_sd

}//namespace f

#endif//JAVMKUPNVXKHKEQRPRFWPJYOCVNHCLMMOAFTBVFQKMSIIGLQSNGJVRXVLVGQTSIFNXOCVKGYX

