#ifndef RTXJTPLUQFORKIANFYMVTINUGMUDPULAXGNYSMATGLFAQPXHMXLSJGBQVQUUVYTWXCMKQJOWM
#define RTXJTPLUQFORKIANFYMVTINUGMUDPULAXGNYSMATGLFAQPXHMXLSJGBQVQUUVYTWXCMKQJOWM

#include <f/pattern/xpattern.hpp>
#include <f/cuda_pattern/cuda_xpattern_config.hpp>
#include <f/cuda_pattern/cuda_xpattern_data.hpp>
#include <f/host/cuda/cuda.hpp>
#include <f/matrix/matrix.hpp>

#include <vector>
#include <functional>

#include <iterator>
#include <algorithm>
#include <iostream>
#include <iomanip>

void make_pattern_intensity_diff( double* cuda_ug, unsigned long* cuda_ar, double* cuda_diag, double thickness, unsigned long* cuda_dim,
                                  double* cuda_I_exp, double* cuda_I_diff, unsigned long column_index, double2* cuda_cache, unsigned long tilt_size, unsigned long max_dim );

//TODO:
//
void make_xpattern_intensity_diff( double* cuda_ug, unsigned long* cuda_ar, double* cuda_diag, double* cuda_thickness, unsigned long* cuda_dim,
                                   double* cuda_I_exp, double* cuda_I_diff, unsigned long column_index, double2* cuda_cache, unsigned long tilt_size, unsigned long max_dim );

namespace f
{

    struct cuda_xpattern
    {
        typedef double                  value_type;
        typedef unsigned long int       size_type;

        cuda_xpattern_config            config;
        cuda_xpattern_data              data;

        size_type unknowns()
        {
            return config.ug_size + config.ug_size + config.tilt_size;
        }

        void set_device()
        {
            int current_id;
            cuda_assert( cudaGetDevice(&current_id) );
            if ( current_id != config.device_id ) cuda_assert( cudaSetDevice( config.device_id ) );
        }

        cuda_xpattern( xpattern<value_type> const& pat, int device_id = 0 ) : config{ make_cuda_xpattern_config( pat, device_id ) }, data{ config }
        {
            int current_id;
            cuda_assert( cudaGetDevice(&current_id) );
            if ( current_id != config.device_id ) cuda_assert( cudaSetDevice( config.device_id ) );

            import( pat );
        }

        //
        std::function<value_type(value_type*)> make_merit_function()
        {
            return [this]( value_type* p )
            {
                return (*this).square_residual( p );
            };
        }

        //
        std::function<value_type(value_type*)> make_abs_function()
        {
            return [this]( value_type* p )
            {
                return (*this).abs_residual( p );
            };
        }

        //
        value_type square_residual( value_type* ug )
        {
            set_device();

            update_I_diff(ug);

            value_type residual;
            cublasHandle_t handle;
            cublas_assert( cublasCreate_v2(&handle) );
            cublas_assert( cublasDdot_v2( handle, static_cast<int>(config.max_dim*config.tilt_size), data.I_diff, 1, data.I_diff, 1, &residual ) );
            cublas_assert( cublasDestroy_v2(handle) );
            return residual;
        }

        //
        value_type abs_residual( value_type* ug )
        {
            set_device();

            update_I_diff(ug);

            value_type residual;
            cublasHandle_t handle;
            cublas_assert( cublasCreate_v2(&handle) );
            cublas_assert( cublasDasum_v2( handle, static_cast<int>(config.max_dim*config.tilt_size), data.I_diff, 1, &residual ) );
            cublas_assert( cublasDestroy_v2(handle) );

            return residual;
        }

        //
        void import( xpattern<value_type> const& pat )
        {
            std::vector<size_type> v_dims;

            for ( size_type index = 0; index != config.tilt_size; ++index )
            {
                v_dims.push_back( pat.diag[index].size() );
                //ar
                size_type const ar_offset = index * config.max_dim * config.max_dim;
                host_to_device_copy_n( pat.ar[index].data(), pat.ar[index].size(), data.ar+ar_offset );
                //diag
                size_type const diag_offset = index * config.max_dim;
                host_to_device_copy_n( pat.diag[index].data(), pat.diag[index].size(), data.diag+diag_offset );
                //intensity
                size_type const I_exp_offset = index * config.max_dim;
                host_to_device_copy_n( pat.intensity[index].data(), pat.intensity[index].size(), data.I_exp+I_exp_offset );
            }

            host_to_device_copy_n( v_dims.data(), v_dims.size(), data.dim );
            host_to_device_copy_n( pat.thickness_array.data(), pat.thickness_array.size(), data.thickness_array );
        }

        void update_I_diff( value_type* x )
        {
            update_ug( x );
            size_type const offset = config.ug_size + config.ug_size;
            update_thickness( x + offset );

            make_xpattern_intensity_diff( data.ug, data.ar, data.diag, data.thickness_array, data.dim, data.I_exp, data.I_diff, config.column_index, data.cache, config.tilt_size, config.max_dim );
        }

        void update_thickness( value_type* thickness )
        {
            host_to_device_copy_n( thickness, config.tilt_size, data.thickness_array );
        }

        void update_ug( value_type* ug )
        {
            host_to_device_copy_n( ug, config.ug_size+config.ug_size, data.ug );
        }

        //dumps
        void dump_ug()
        {
            set_device();

            matrix<double> ug{ 1, config.ug_size*2 };
            device_to_host_copy_n( data.ug, ug.size(), &ug[0][0] );
            std::cout << "\nUg is " << ug.transpose() << "\n";
        }

        void dump_a()
        {
            set_device();

            matrix<double> a{ config.max_dim, config.max_dim+config.max_dim };
            device_to_host_copy_n( data.cache, a.size(), &a[0][0] );
            std::cout << "a is \n";
            std::cout << a << "\n";
        }

        void dump_I_diff()
        {
            set_device();

            matrix<double> Idiff{ config.tilt_size, config.max_dim };
            device_to_host_copy_n( data.I_diff, Idiff.size(), &Idiff[0][0] );

            std::cout.precision( 15 );
            std::cout << Idiff << "\n";
            std::cout << "\nsquare residual is " << std::inner_product( Idiff.begin(), Idiff.end(), Idiff.begin(), 0.0 ) << "\n";
        }

        void dump_I_exp()
        {
            set_device();

            matrix<double> Iexp{ config.tilt_size, config.max_dim };
            //cuda_assert( cudaMemcpy( reinterpret_cast<void*>(Iexp.data()), reinterpret_cast<const void*>(data.I_exp), sizeof(value_type) * Iexp.size(), cudaMemcpyDeviceToHost ) );
            device_to_host_copy_n( data.I_exp, Iexp.size(), Iexp.data() ); 
            std::copy( Iexp.row_begin(0), Iexp.row_end(), std::ostream_iterator<double>( std::cout, "\t" ) );
            std::cout << "\n";
        }

    };//struct cuda_xpattern

}

#endif//RTXJTPLUQFORKIANFYMVTINUGMUDPULAXGNYSMATGLFAQPXHMXLSJGBQVQUUVYTWXCMKQJOWM

