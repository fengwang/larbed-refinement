#ifndef VVJJASVJBTWGCYGGIKTOFPRKJVOERMHOPIPWHXWOLFEPDGQGQJNFPTCGFATCKXMMHQCMUYPCS
#define VVJJASVJBTWGCYGGIKTOFPRKJVOERMHOPIPWHXWOLFEPDGQGQJNFPTCGFATCKXMMHQCMUYPCS

#include <f/device/device_matrix/device_matrix.hpp>
#include <f/matrix/matrix.hpp>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cassert>
#include <iostream>
#include <complex>

namespace f
{
 
    struct device_matrix<std::complex<float>>
    {
        typedef float2                      value_type;
        typedef value_type*                 pointer;
        typedef unsigned long               size_type;
        typedef device_matrix               self_type;

        device_matrix( size_type const row, size_type const col, void* host_data ) : row_(row), col_(col)
        {
            unsigned long const size = host.size();
            cuda_assert( cudaMalloc( (void **)&data_, size * sizeof( value_type ) ) );
            cuda_assert( cudaMemcpy( data_, host_data, size * sizeof( value_type ), cudaMemcpyHostToDevice ) );
        }

        device_matrix( matrix<std::complex<float>> const& host ) : row_( host.row() ), col_( host.col() )
        {
            unsigned long const size = host.size();
            cuda_assert( cudaMalloc( (void **)&data_, size * sizeof( value_type ) ) );
            cuda_assert( cudaMemcpy( data_, host.data(), size * sizeof( value_type ), cudaMemcpyHostToDevice ) );
        }

        device_matrix( matrix<float> const& host_real, matrix<float> const& host_imag )
        {
            assert( host_real.row() == host_imag.row() );
            assert( host_real.col() == host_imag.col() );

            row_ = host_real.row();
            col_ = host_real.col();

            unsigned long const size = row_ * col_;
            cuda_assert( cudaMalloc( (void **)&data_, size * sizeof( value_type ) ) );

            //implemented in file 'src/device/kernel/device_matrix/cmatrix_memcpy_real_imag_impl.cu'
            void cmatrix_memcpy_real_imag_impl( float2*, float*, float*, unsigned long const );
            cmatrix_memcpy_real_imag_impl( data_, host_real.data(), host_imag.data(), size );
        }

        //constructor with real and imag part respectively
        device_matrix( size_type const row, size_type const col, float* host_data_real, float* host_data_imag ) : row_(row), col_(col)
        {
            unsigned long const size = row * col;
            cuda_assert( cudaMalloc( (void **)&data_, size* sizeof( value_type ) ) );

            //implemented in file 'src/device/kernel/device_matrix/cmatrix_memcpy_real_imag_impl.cu'
            void cmatrix_memcpy_real_imag_impl( float2*, float*, float*, unsigned long const );
            cmatrix_memcpy_real_imag_impl( data_, host_data_real, host_data_imag, size );
        }

        device_matrix( size_type const row, size_type const col ) : row_(row), col_(col)
        {
            const size_type total_size = row * col * sizeof( value_type );
            cuda_assert( cudaMalloc( (void **)&data_, total_size ) );
            cuda_assert( cudaMemset( data_, 0, total_size ) );
        }

        device_matrix( const device_matrix& other ) : row_(other.row()), col_(other.col())
        {
            const size_type total_size = other.size() * sizeof( value_type );
            cuda_assert( cudaMalloc( (void **)&data_, total_size ) );
            cuda_assert( cudaMemcpy( data_, other.data_, total_size, cudaMemcpyDeviceToDevice ) );
        }

        self_type& operator = ( const device_matrix& other )
        {
            const size_type total_size = other.size() * sizeof( value_type );
            
            if ( size() != other.size() )
            {
                if( data_ )
                    cuda_assert( cudaFree( data_ ) );

                cuda_assert( cudaMalloc( (void **)&data_, total_size ) );
            }

            row_ = other.row_;
            col_ = other.col_;

            cuda_assert( cudaMemcpy( data_, other.data_, total_size, cudaMemcpyDeviceToDevice ) );

            return *this;
        }

        ~device_matrix()
        {
            row_ = 0;
            col_ = 0;
            cuda_assert( cudaFree( data_ ) );
            data_ = 0;
        }

        void transpose()
        {
            using device_matrix_dsajhio4elkjsansafdioh4ekljansfdkljsanfdlkjnfd::cublas_handle_initializer;
            auto& ci = singleton<cublas_handle_initializer>::instance();

            self_type clone( *this );
            value_type const alpha{ 1.0, 0.0 };
            value_type const beta{ 0.0, 0.0 }; 
            cublas_assert( cublasCgeam( ci.handle, CUBLAS_OP_T, CUBLAS_OP_N, row(), col(), &alpha, clone.data(), clone.col(), &beta, clone.data(), clone.row(), data(), row()  ) );

            row_ = clone.col();
            col_ = clone.row();
        }

        float norm() const
        {
            using device_matrix_dsajhio4elkjsansafdioh4ekljansfdkljsanfdlkjnfd::cublas_handle_initializer;
            auto& ci = singleton<cublas_handle_initializer>::instance();

            float result = 0;
            
            cublas_assert( cublasScasum( ci.handle, size(), (const cuComplex*)data(), 1,  &result ) );

            return result;
        }

        float sim_norm_1() const
        {
            return norm() / col();
        }
        
        void export_to( void* host_data ) const
        {
            const size_type total_size = row() * col() * sizeof( value_type );
            cuda_assert( cudaMemcpy( host_data, data_, total_size, cudaMemcpyDeviceToHost ) );
        }
        
        size_type size() const
        {
            return row() * col();
        }

        friend self_type const operator + ( const self_type& lhs, const self_type& rhs )
        {
            assert( lhs.row() == rhs.row() );
            assert( lhs.col() == rhs.col() );

            using device_matrix_dsajhio4elkjsansafdioh4ekljansfdkljsanfdlkjnfd::cublas_handle_initializer;
            auto& ci = singleton<cublas_handle_initializer>::instance();

            self_type ans( lhs );
           
            value_type const alpha{ 1.0, 0.0 };
            cublas_assert( cublasCaxpy( ci.handle, ans.row()*ans.col(), &alpha, rhs.data(), 1, ans.data(), 1 ) ); // ans = lhs + rhs
            
            return ans;
        }

        friend self_type const operator - ( const self_type& lhs, const self_type& rhs )
        {
            assert( lhs.row() == rhs.row() );
            assert( lhs.col() == rhs.col() );

            using device_matrix_dsajhio4elkjsansafdioh4ekljansfdkljsanfdlkjnfd::cublas_handle_initializer;
            auto& ci = singleton<cublas_handle_initializer>::instance();

            self_type ans( lhs );
           
            value_type const alpha{ -1.0, 0.0 };
            cublas_assert( cublasCaxpy( ci.handle, lhs.row()*lhs.col(), &alpha, rhs.data(), 1, ans.data(), 1 ) ); // ans = lhs - rhs 
            
            return ans;
        }

        friend self_type const operator / ( const self_type& lhs, const float rhs )
        {
            return lhs * (1.0/rhs);
        }

        friend self_type const operator * ( const self_type& lhs, const float rhs )
        {
            using device_matrix_dsajhio4elkjsansafdioh4ekljansfdkljsanfdlkjnfd::cublas_handle_initializer;
            auto& ci = singleton<cublas_handle_initializer>::instance();

            self_type ans( lhs );
           
            value_type const alpha{ (float)(rhs-1.0), (float)0.0 };
            cublas_assert( cublasCaxpy( ci.handle, ans.row()*ans.col(), &alpha, ans.data(), 1, ans.data(), 1 ) ); //  ans = (alpha-1) ans + ans = alpha ans
            
            return ans;
        }

        friend self_type const operator * ( const float lhs, const self_type& rhs )
        {
            return rhs * lhs;
        }

        friend self_type const operator * ( const self_type& lhs, const self_type& rhs )
        {
            assert( lhs.col() == rhs.row() );

            using device_matrix_dsajhio4elkjsansafdioh4ekljansfdkljsanfdlkjnfd::cublas_handle_initializer;
            auto& ci = singleton<cublas_handle_initializer>::instance();

            self_type ans( lhs.row(), rhs.col() ); //ans = {0, ....., 0}

            const unsigned long m = lhs.row();
            const unsigned long k = lhs.col();
            const unsigned long n = rhs.col();
            value_type const alpha{ 1.0, 0.0 };
            value_type const beta{ 0.0, 0.0 };

            cublas_assert( cublasCgemm( ci.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, rhs.data(), n, lhs.data(), k, &beta, ans.data(), n ) );
            
            return ans;
        }

        friend std::ostream& operator << ( std::ostream& lhs, const self_type& rhs )
        {
            const unsigned long total_size = rhs.row() * rhs.col() * sizeof(value_type);
            pointer data = new value_type[ rhs.row() * rhs.col() ];

            cuda_assert( cudaMemcpy( (void*)data, (const void*)rhs.data(), total_size, cudaMemcpyDeviceToHost ) );

            for ( unsigned long r = 0; r != rhs.row(); ++r )
            {
                for ( unsigned long c = 0; c != rhs.col(); ++c )
                {
                    lhs << "(" << data[r*rhs.col()+c].x << ", " << data[r*rhs.col()+c].y << ")\t";
                }

                lhs << "\n";
            }

            delete[] data;

            return lhs;
        }

        size_type row() const
        {
            return row_;
        }

        size_type col() const
        {
            return col_;
        }

        pointer data() const
        {
            return data_;
        }

        pointer data()
        {
            return data_;
        }

        size_type row_;
        size_type col_;
        pointer data_;

    };//struct device_matrix

}//namespace f

#endif//VVJJASVJBTWGCYGGIKTOFPRKJVOERMHOPIPWHXWOLFEPDGQGQJNFPTCGFATCKXMMHQCMUYPCS

