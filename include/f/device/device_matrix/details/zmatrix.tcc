#ifndef SDOPNASFKLJASDIOU4HEFAUHKLASJH498YUHAFDIOJUHASFKLJH49HUASFDKJBNVAFHDSAIUF
#define SDOPNASFKLJASDIOU4HEFAUHKLASJH498YUHAFDIOJUHASFKLJH49HUASFDKJBNVAFHDSAIUF

#include <f/device/device_matrix/device_matrix.hpp>
#include <f/matrix/matrix.hpp>
#include <f/device/utility/cublas_handle.hpp>
#include <f/singleton/singleton.hpp>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cassert>
#include <iostream>
#include <complex>

//implemented in file 'src/device/kernel/device_matrix/zmatrix_memcpy_real_imag_impl.cu'
void zmatrix_memcpy_real_imag_impl( double2*, const double*, const double*, unsigned long const );

//implemented in file 'src/device/kernel/device_matrix/zmatrix_eye_impl.cu'
void zmatrix_eye_impl( double2*, unsigned long const, double );

void zmatrix_eye_impl( double2*, unsigned long const, double real, double imag );

namespace f
{
    template<> 
    struct device_matrix<std::complex<double>>
    {
        typedef double2                     value_type;
        typedef value_type*                 pointer;
        typedef unsigned long               size_type;
        typedef device_matrix               self_type;

        //diag matrix with initial value 'val'
        device_matrix( size_type const n, double val ) : row_( n ), col_( n )
        {
            unsigned long const size = n*n;
            cuda_assert( cudaMalloc( (void **)&data_, size * sizeof( value_type ) ) );
            //cuda_assert( cudaMemcpy( data_, host_data, size * sizeof( value_type ), cudaMemcpyHostToDevice ) );
            cuda_assert( cudaMemset( data_, 0, size*sizeof(value_type) ) );
            zmatrix_eye_impl( data(), n, val );
        }

        //diag matrix with initial value 'real, imag'
        device_matrix( size_type const n, double real, double imag ) : row_( n ), col_( n )
        {
            unsigned long const size = n*n;
            cuda_assert( cudaMalloc( (void **)&data_, size * sizeof( value_type ) ) );
            cuda_assert( cudaMemset( data_, 0, size*sizeof(value_type) ) );
            zmatrix_eye_impl( data(), n, real, imag );
        }

        device_matrix( size_type const row, size_type const col, void* host_data ) : row_(row), col_(col)
        {
            unsigned long const size = row * col;
            cuda_assert( cudaMalloc( (void **)&data_, size * sizeof( value_type ) ) );
            cuda_assert( cudaMemcpy( data_, host_data, size * sizeof( value_type ), cudaMemcpyHostToDevice ) );
        }

        device_matrix( matrix<std::complex<double>> const& host ) : row_( host.row() ), col_( host.col() )
        {
            unsigned long const size = host.size();
            cuda_assert( cudaMalloc( (void **)&data_, size * sizeof( value_type ) ) );
            cuda_assert( cudaMemcpy( data_, host.data(), size * sizeof( value_type ), cudaMemcpyHostToDevice ) );
        }

        device_matrix( matrix<double> const& host_real, matrix<double> const& host_imag )
        {
            assert( host_real.row() == host_imag.row() );
            assert( host_real.col() == host_imag.col() );

            row_ = host_real.row();
            col_ = host_real.col();

            unsigned long const size = row_ * col_;
            cuda_assert( cudaMalloc( (void **)&data_, size * sizeof( value_type ) ) );

            zmatrix_memcpy_real_imag_impl( data_, host_real.data(), host_imag.data(), size );
        }

        //constructor with real and imag part respectively
        device_matrix( size_type const row, size_type const col, double* host_data_real, double* host_data_imag ) : row_(row), col_(col)
        {
            unsigned long const size = row * col;
            cuda_assert( cudaMalloc( (void **)&data_, size* sizeof( value_type ) ) );
            zmatrix_memcpy_real_imag_impl( data_, host_data_real, host_data_imag, size );
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
            auto& ci = singleton<cublas_handle>::instance();

            self_type clone( *this );
            value_type const alpha{ 1.0, 0.0 };
            value_type const beta{ 0.0, 0.0 }; 
            cublas_assert( cublasZgeam( ci.handle, CUBLAS_OP_T, CUBLAS_OP_N, row(), col(), &alpha, clone.data(), clone.col(), &beta, clone.data(), clone.row(), data(), row()  ) );

            row_ = clone.col();
            col_ = clone.row();
        }

        double norm() const
        {
            auto& ci = singleton<cublas_handle>::instance();

            double result = 0;
            
            cublas_assert( cublasDzasum( ci.handle, size(), (const cuDoubleComplex*)data(), 1,  &result ) );

            return result;
        }

        double sim_norm_1() const
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

        self_type& operator += ( const self_type& other )
        {
            assert( row() == other.row() );
            assert( col() == other.col() );

            auto& ci = singleton<cublas_handle>::instance();
            value_type const alpha{ 1.0, 0.0 };

            cublas_assert( cublasZaxpy( ci.handle, size(), &alpha, other.data(), 1, data(), 1 ) ); // ans = lhs + rhs
            return *this;
        }

        friend self_type const operator + ( const self_type& lhs, const self_type& rhs )
        {
            self_type ans( lhs );
            ans += rhs;
            return ans;
            /*
            assert( lhs.row() == rhs.row() );
            assert( lhs.col() == rhs.col() );

            auto& ci = singleton<cublas_handle>::instance();

            self_type ans( lhs );
           
            value_type const alpha{ 1.0, 0.0 };
            cublas_assert( cublasZaxpy( ci.handle, ans.row()*ans.col(), &alpha, rhs.data(), 1, ans.data(), 1 ) ); // ans = lhs + rhs
            
            return ans;
            */
        }

        self_type& operator -= ( const self_type& other )
        {
            assert( row() == other.row() );
            assert( col() == other.col() );

            auto& ci = singleton<cublas_handle>::instance();
            value_type const alpha{ -1.0, 0.0 };

            cublas_assert( cublasZaxpy( ci.handle, size(), &alpha, other.data(), 1, data(), 1 ) ); // ans = lhs + rhs
            return *this;
        }

        friend self_type const operator - ( const self_type& lhs, const self_type& rhs )
        {
            self_type ans( lhs );
            ans -= rhs;
            return ans;
            /*
            assert( lhs.row() == rhs.row() );
            assert( lhs.col() == rhs.col() );

            auto& ci = singleton<cublas_handle>::instance();

            self_type ans( lhs );
           
            value_type const alpha{ -1.0, 0.0 };
            cublas_assert( cublasZaxpy( ci.handle, lhs.row()*lhs.col(), &alpha, rhs.data(), 1, ans.data(), 1 ) ); // ans = lhs - rhs 
            
            return ans;
            */
        }

        self_type& operator *= ( const std::complex<double>& other )
        {
            auto& ci = singleton<cublas_handle>::instance();
            value_type const alpha{ (double)(std::real(other)-1.0), std::imag(other) };
            cublas_assert( cublasZaxpy( ci.handle, size(), &alpha, data(), 1, data(), 1 ) ); 
        
            return *this;
        }

        self_type& operator *= ( const double other )
        {
            return operator *= ( std::complex<double>{other, 0.0} );
        }

        friend self_type const operator * ( const self_type& lhs, const double rhs )
        {
            self_type ans( lhs );
            ans *= rhs;
            return ans;
        }

        friend self_type const operator * ( const double lhs, const self_type& rhs )
        {
            return rhs * lhs;
        }

        friend self_type const operator * ( const self_type& lhs, const self_type& rhs )
        {
            assert( lhs.col() == rhs.row() );

            auto& ci = singleton<cublas_handle>::instance();

            self_type ans( lhs.row(), rhs.col() ); //ans = {0, ....., 0}

            const unsigned long m = lhs.row();
            const unsigned long k = lhs.col();
            const unsigned long n = rhs.col();
            value_type const alpha{ 1.0, 0.0 };
            value_type const beta{ 0.0, 0.0 };

            cublas_assert( cublasZgemm( ci.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, rhs.data(), n, lhs.data(), k, &beta, ans.data(), n ) );
            
            return ans;
        }

        friend self_type const operator / ( const self_type& lhs, const double rhs )
        {
            return lhs * (1.0/rhs);
        }
/*
        void to_eye()
        {
            assert( row() == col() );
            cm_eye( data(), row() );
        }
        */
         
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

#endif//VCHJNACHOIXDKXSXONEJMPWDPBXTSHXNDHVAFJAVVYVRSWQPYGNMVPRHYYMNURHRLYHSFVGXO

