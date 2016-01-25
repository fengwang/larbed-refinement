#ifndef SJNSKDJASDU8HY4KLJDFHSLKAJSDFH4UY7HASLDKHJUASLDKFHJ49UYHASFDLHJASFKJHSFI9
#define SJNSKDJASDU8HY4KLJDFHSLKAJSDFH4UY7HASLDKHJUASLDKFHJ49UYHASFDLHJASFKJHSFI9

#include <f/pattern/pattern.hpp>
#include <f/cuda_pattern/cuda_pattern_config.hpp>
#include <f/cuda_pattern/cuda_pattern_data.hpp>
#include <f/host/cuda/cuda.hpp>
#include <f/matrix/matrix.hpp>

#include <vector>
#include <functional>
#include <cmath>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <iomanip>

void make_pattern_intensity_diff( double* cuda_ug, unsigned long* cuda_ar, double* cuda_diag, double thickness, unsigned long* cuda_dim, double* cuda_I_exp, double* cuda_I_diff, unsigned long column_index, double2* cuda_cache, unsigned long tilt_size, unsigned long max_dim, double c, double* cuda_I_zigmoid, double* beams, double* kt_factor );

namespace f
{

    struct cuda_pattern
    {
        typedef double                  value_type;
        typedef unsigned long int       size_type;

        cuda_pattern_config             config;
        cuda_pattern_data               data;
        std::vector<value_type>         ug_norm_cache;
        matrix<double>                  sim_ug; //[n][3]

        size_type unknowns() const
        {
            return config.tilt_size*3+1;
        }

        void update_ug()
        {
            matrix<double> sim_ug_( sim_ug.row(), 2 );
            std::copy( sim_ug.col_begin(1), sim_ug.col_end(1), sim_ug_.col_begin(0) );
            std::fill( sim_ug.col_begin(2), sim_ug.col_end(2), 0.0 );
            cuda_assert( cudaMemcpy( reinterpret_cast<void*>(data.ug), reinterpret_cast<const void*>(sim_ug_.data()), config.ug_size*sizeof(value_type)*2, cudaMemcpyHostToDevice ) );
        }

        cuda_pattern( pattern<value_type> const& pat, int device_id, std::string const& ug_path ) : config{ make_cuda_pattern_config( pat, device_id ) }, data{ config }
        {
            int current_id;
            cuda_assert( cudaGetDevice(&current_id) );
            if ( current_id != config.device_id ) cuda_assert( cudaSetDevice( config.device_id ) );

            import( pat );

            sim_ug.load( ug_path );
            update_ug();
        }

        std::function<value_type(value_type*)> make_merit_function()
        {
            return [this]( value_type* p )
            {
                value_type* ug = p;
                value_type thickness = *(p+(*this).config.ug_size*3);
                return (*this).square_residual( ug, thickness );
            };
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

        std::function<value_type(value_type*)> make_abs_function()
        {
            return [this]( value_type* p )
            {
                value_type* ug = p;
                value_type thickness = *(p+(*this).config.ug_size*3);
                return (*this).abs_residual( ug, thickness );
            };
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

        void import( pattern<value_type> const& pat )
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

            cuda_assert( cudaMemcpy( reinterpret_cast<void*>(data.dim), reinterpret_cast<const void*>(v_dims.data()), sizeof(size_type) * v_dims.size(), cudaMemcpyHostToDevice ) );
        }

        void update_I_diff( value_type* kt, value_type thickness, value_type c = 0.0 )
        {
            update_thickness( thickness );

            update_kt_factor( kt );

            make_pattern_intensity_diff( data.ug, data.ar, data.diag, config.thickness, data.dim, data.I_exp, data.I_diff, config.column_index, data.cache, config.tilt_size, config.max_dim, c, data.I_zigmoid, data.beams, data.kt_factor );
        }

        void update_kt_factor( value_type* host_kt_factor )
        {
            cuda_assert( cudaMemcpy( reinterpret_cast<void*>(data.kt_factor), reinterpret_cast<const void*>(host_kt_factor), config.tilt_size*sizeof(value_type)*3, cudaMemcpyHostToDevice ) );
        }

        void update_thickness( value_type thickness )
        {
            config.thickness = thickness;
        }

    };//struct cuda_pattern

}

#endif

