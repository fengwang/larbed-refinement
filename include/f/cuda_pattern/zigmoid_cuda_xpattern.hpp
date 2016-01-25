#ifndef XGIXHAYEUUKPPMJGCRPXLTVECPKMHVBEUYQPHEPAWXABSIGVJWJEQYRSLJHQSUOTXMPOSBAGH
#define XGIXHAYEUUKPPMJGCRPXLTVECPKMHVBEUYQPHEPAWXABSIGVJWJEQYRSLJHQSUOTXMPOSBAGH

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

void make_pattern_intensity_diff( double* cuda_ug, unsigned long* cuda_ar, double* cuda_diag, double thickness, unsigned long* cuda_dim, double* cuda_I_exp, double* cuda_I_diff, unsigned long column_index, double2* cuda_cache, unsigned long tilt_size, unsigned long max_dim, double c, double* cuda_I_zigmoid );

namespace f
{

    struct cuda_pattern
    {
        typedef double                  value_type;
        typedef unsigned long int       size_type;

        cuda_xpattern_config            config;
        cuda_xpattern_data              data;

        cuda_pattern( xpattern<value_type> const& pat, int device_id = 0 ) : config{ make_cuda_xpattern_config( pat, device_id ) }, data{ config }
        {
            int current_id;
            cuda_assert( cudaGetDevice(&current_id) );
            if ( current_id != config.device_id ) cuda_assert( cudaSetDevice( config.device_id ) );

            import( pat );
        }

        std::function<value_type(value_type*)> make_zigmoid_function( value_type c )
        {
            return [this, c]( value_type* p )
            {
                value_type* ug = p;
                value_type thickness = *(p+(*this).config.ug_size*2);
                return (*this).zigmoid_residual( ug, thickness, c );
            };
        }

        std::function<value_type(value_type*)> make_merit_function()
        {
            return [this]( value_type* p )
            {
                value_type* ug = p;
                value_type thickness = *(p+(*this).config.ug_size*2);
                return (*this).square_residual( ug, thickness );
            };
        }

        std::function<value_type(value_type*)> make_abs_function()
        {
            return [this]( value_type* p )
            {
                value_type* ug = p;
                value_type thickness = *(p+(*this).config.ug_size*2);
                return (*this).abs_residual( ug, thickness );
            };
        }


    private:

        value_type zigmoid_residual( value_type* ug, value_type thickness, value_type c )
        {
            int current_id;
            cuda_assert( cudaGetDevice(&current_id) );
            if ( current_id != config.device_id ) cuda_assert( cudaSetDevice( config.device_id ) );

            update_I_diff(ug, thickness, c);

            value_type residual;
            cublasHandle_t handle;
            cublas_assert( cublasCreate_v2(&handle) );
            cublas_assert( cublasDdot_v2( handle, static_cast<int>(config.max_dim*config.tilt_size), data.I_zigmoid, 1, data.I_zigmoid, 1, &residual ) );
            cublas_assert( cublasDestroy_v2(handle) );
            return residual;
        }

        value_type square_residual( value_type* ug, value_type thickness )
        {
            int current_id;
            cuda_assert( cudaGetDevice(&current_id) );
            if ( current_id != config.device_id ) cuda_assert( cudaSetDevice( config.device_id ) );

            update_I_diff(ug, thickness);

            value_type residual;
            cublasHandle_t handle;
            cublas_assert( cublasCreate_v2(&handle) );
            cublas_assert( cublasDdot_v2( handle, static_cast<int>(config.max_dim*config.tilt_size), data.I_diff, 1, data.I_diff, 1, &residual ) );
            cublas_assert( cublasDestroy_v2(handle) );
            return residual;
        }

        value_type abs_residual( value_type* ug, value_type thickness )
        {
            int current_id;
            cuda_assert( cudaGetDevice(&current_id) );
            if ( current_id != config.device_id ) cuda_assert( cudaSetDevice( config.device_id ) );

            update_I_diff(ug, thickness);

            value_type residual;
            cublasHandle_t handle;
            cublas_assert( cublasCreate_v2(&handle) );
            cublas_assert( cublasDasum_v2( handle, static_cast<int>(config.max_dim*config.tilt_size), data.I_diff, 1, &residual ) );
            cublas_assert( cublasDestroy_v2(handle) );

            return residual;
        }

        void import( xpattern<value_type> const& pat )
        {
            std::vector<size_type> v_dims;

            for ( size_type index = 0; index != config.tilt_size; ++index )
            {
                v_dims.push_back( pat.diag[index].size() );
                //ar
                size_type const ar_offset = index * config.max_dim * config.max_dim;
                size_type const ar_size = pat.ar[index].size() * sizeof( size_type );
                cuda_assert( cudaMemcpy( reinterpret_cast<void*>(data.ar + ar_offset), reinterpret_cast<const void*>(pat.ar[index].data()), ar_size, cudaMemcpyHostToDevice ) );
                //diag
                size_type const diag_offset = index * config.max_dim;
                size_type const diag_size = pat.diag[index].size() * sizeof( value_type );
                cuda_assert( cudaMemcpy( reinterpret_cast<void*>(data.diag + diag_offset), reinterpret_cast<const void*>(pat.diag[index].data()), diag_size, cudaMemcpyHostToDevice ) );
                //intensity
                size_type const I_exp_offset = index * config.max_dim;
                size_type const I_exp_size = pat.intensity[index].size() * sizeof( value_type );
                cuda_assert( cudaMemcpy( reinterpret_cast<void*>(data.I_exp + I_exp_offset), reinterpret_cast<const void*>(pat.intensity[index].data()), I_exp_size, cudaMemcpyHostToDevice ) );
            }

            cuda_assert( cudaMemcpy( reinterpret_cast<void*>(data.thickness_array), reinterpret_cast<const void*>(pat.thickness_array.data()), sizeof(value_type)*pat.tilt_size, cudaMemcpyHostToDevice ) );

            cuda_assert( cudaMemcpy( reinterpret_cast<void*>(data.dim), reinterpret_cast<const void*>(v_dims.data()), sizeof(size_type) * v_dims.size(), cudaMemcpyHostToDevice ) );
        }

        void update_I_diff( value_type* ug, value_type thickness, value_type c = 0.0 )
        {
            update_thickness( thickness );
            update_ug( ug );

            make_pattern_intensity_diff( data.ug, data.ar, data.diag, config.thickness, data.dim, data.I_exp, data.I_diff, config.column_index, data.cache, config.tilt_size, config.max_dim, c, data.I_zigmoid );
        }

        void update_thickness( value_type thickness )
        {
            config.thickness = thickness;
        }

        void update_ug( value_type* ug )
        {
            cuda_assert( cudaMemcpy( reinterpret_cast<void*>(data.ug), reinterpret_cast<const void*>(ug), config.ug_size*sizeof(value_type)*2, cudaMemcpyHostToDevice ) );
        }

    };//struct cuda_pattern

}

#endif//XGIXHAYEUUKPPMJGCRPXLTVECPKMHVBEUYQPHEPAWXABSIGVJWJEQYRSLJHQSUOTXMPOSBAGH

