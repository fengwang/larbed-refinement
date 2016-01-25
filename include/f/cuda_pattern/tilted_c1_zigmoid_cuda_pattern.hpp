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

        cuda_pattern( pattern<value_type> const& pat, int device_id = 0 ) : config{ make_cuda_pattern_config( pat, device_id ) }, data{ config }
        {
            int current_id;
            cuda_assert( cudaGetDevice(&current_id) );
            if ( current_id != config.device_id ) cuda_assert( cudaSetDevice( config.device_id ) );

            import( pat );

            make_ug_norm( pat );
        }

        std::function<value_type(value_type*)> make_c1_abs_constrain_zigmoid_function( value_type c, value_type lambda, value_type beta )
        {
            return [this, c, lambda, beta]( value_type* p )
            {
                auto zig = (*this).make_zigmoid_function( c );
                auto c1 = (*this).make_c1_function( lambda );
                //TODO
                //auto constrain = (*this).make_abs_constrain_function( beta );
                auto constrain = (*this).make_zero_constrain_function( beta );
                return zig( p ) + c1( p ) + constrain( p );
            };
        }

        std::function<value_type(value_type*)> make_c1_constrain_zigmoid_function( value_type c, value_type lambda, value_type beta )
        {
            return [this, c, lambda, beta]( value_type* p )
            {
                auto zig = (*this).make_zigmoid_function( c );
                auto c1 = (*this).make_c1_function( lambda );
                auto constrain = (*this).make_zero_constrain_function( beta );
                return zig( p ) + c1( p ) + constrain( p );
            };
        }

        std::function<value_type(value_type*)> make_c1_zigmoid_function( value_type c, value_type lambda )
        {
            return [this, c, lambda]( value_type* p )
            {
                auto zig = (*this).make_zigmoid_function( c );
                auto c1 = (*this).make_c1_function( lambda );
                return zig( p ) + c1( p );
            };
        }

        std::function<value_type(value_type*)> make_abs_constrain_function( value_type beta )
        {
            return [this,beta]( value_type* p )
            {
                value_type const eps = 1.0e-20;
                value_type abs_res{0};
                for ( size_type index = 1; index != (*this).ug_norm_cache.size(); ++index )
                {
                    if ( ((*this).ug_norm_cache)[index] > eps ) continue;
                    value_type const real = p[index+index];
                    value_type const imag = p[index+index+1];
                    abs_res += std::sqrt( real*real+imag*imag+eps );
                }
                return abs_res*beta;
            };
        }

        std::function<value_type(value_type*)> make_zero_constrain_function( value_type beta )
        {
            return [this,beta]( value_type* p )
            {
                value_type res{0.0};
                for ( size_type index = 1; index != (*this).ug_norm_cache.size(); ++index )
                {
                    value_type const eps{1.0e-10};
                    if ( ((*this).ug_norm_cache)[index] > eps ) continue;
                    value_type const real = p[index+index];
                    value_type const imag = p[index+index+1];
                    res += real*real+imag*imag;
                }
                return res*beta;
            };
        }

        std::function<value_type(value_type*)> make_c1_function( value_type lambda )
        {
            return [this,lambda]( value_type* p )
            {
                value_type res{ 0 };

                for ( size_type index = 0; index != (*this).ug_norm_cache.size(); ++index )
                {
                    value_type norm = (*this).ug_norm_cache[index];
                    if ( norm > value_type{0} )
                    {
                        size_type const r_index = index+index;
                        size_type const i_index = r_index+1;
                        value_type const df = std::hypot( p[r_index], p[i_index] ) - norm;
                        res += df * df;
                    }
                }

                return lambda * res;
            };
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

        void import_beams( std::string const& path )
        {
            matrix<double> host_beams;
            host_beams.load( path );
            cuda_assert( cudaMemcpy( reinterpret_cast<void*>(data.beams), reinterpret_cast<const void*>(host_beams.data()), sizeof(size_type) * host_beams.size(), cudaMemcpyHostToDevice ) );
        }

        void update_I_diff( value_type* ug, value_type thickness, value_type c = 0.0 )
        {
            update_thickness( thickness );
            update_ug( ug );

            size_type const kt_factor_offset = config.ug_size*2+1;
            update_kt_factor( ug+kt_factor_offset );

            //make_pattern_intensity_diff( data.ug, data.ar, data.diag, config.thickness, data.dim, data.I_exp, data.I_diff, config.column_index, data.cache, config.tilt_size, config.max_dim, c, data.I_zigmoid );
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

        void update_ug( value_type* ug )
        {
            cuda_assert( cudaMemcpy( reinterpret_cast<void*>(data.ug), reinterpret_cast<const void*>(ug), config.ug_size*sizeof(value_type)*2, cudaMemcpyHostToDevice ) );
        }

        void make_ug_norm( pattern<value_type> const & pattern )
        {
            std::map<unsigned long, value_type> cache = extract_ug_norm( pattern );
            ug_norm_cache.resize( pattern.ug_size );
            std::fill( ug_norm_cache.begin(), ug_norm_cache.end(), value_type{-1} );

            for ( auto& elem : cache )
                ug_norm_cache[elem.first] = elem.second;
        }

    };//struct cuda_pattern

}

#endif

