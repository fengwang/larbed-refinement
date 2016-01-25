#ifndef HKFQAYYWBFCGAFYDPRGKQCCFXDYGWOTJORLPTDNFFQHRXLTXBMFNGLLWVMEJEUIVYLYHHNJQQ
#define HKFQAYYWBFCGAFYDPRGKQCCFXDYGWOTJORLPTDNFFQHRXLTXBMFNGLLWVMEJEUIVYLYHHNJQQ

#include <f/device/device_matrix/device_matrix.hpp>

#include <complex>

//implemented in file 'src/device/kernel/device_matrix/zmatrix_eye_impl.cu'
void zmatrix_eye_impl( double2*, unsigned long const );

namespace f
{
    namespace eye_private
    {
        //TODO:
        //      should have a default constructor
        template< typename T >
        struct eye_matrix_builder;

#if 0
        template<>
        struct eye_matrix_builder<float>
        {
            typedef device_matrix<foat>     matrix_type;

            matrix_type const operator()( unsigned long const n );
            {
                //function implemented in file ''
                //TODO: impl this
                extern "C" void smatrix_eye_impl( float*, unsigned long const );
                matrix_type ans( n, n );
                smatrix_eye_impl( ans.data(), n );
                return ans;
            }
        
        };//eye_matrix_builder for smatrix

        template<>
        struct eye_matrix_builder<double>
        {
            typedef device_matrix<double>     matrix_type;

            matrix_type const operator()( unsigned long const n ) const
            {
                //function implemented in file ''
                //TODO: impl this
                extern "C" void dmatrix_eye_impl( double*, unsigned long const );
                matrix_type ans( n, n );
                dmatrix_eye_impl( ans.data(), n );
                return ans;
            }
        
        };//eye_matrix_builder for dmatrix

        template<>
        struct eye_matrix_builder<std::complex<float>>
        {
            typedef device_matrix<std::complex<float>>     matrix_type;

            matrix_type const operator()( unsigned long const n ) const
            {
                //function implemented in file ''
                //TODO: impl this
                extern "C" void cmatrix_eye_impl( float2*, unsigned long const );
                matrix_type ans( n, n );
                cmatrix_eye_impl( ans.data(), n );
                return ans;
            }
        
        };//eye_matrix_builder for cmatrix
#endif

        template<>
        struct eye_matrix_builder<std::complex<double>>
        {
            typedef device_matrix<std::complex<double>>     matrix_type;

            matrix_type const operator()( unsigned long const n ) const
            {
                matrix_type ans( n, n );
                zmatrix_eye_impl( ans.data(), n );
                return ans;
            }
        
        };//eye_matrix_builder for zmatrix

    }//namespace eye_private


    template<typename T>
    device_matrix<T> const device_eye( const unsigned long n )
    {
        return eye_private::eye_matrix_builder<T>()( n );
    }
    
}//namespace f

#endif//HKFQAYYWBFCGAFYDPRGKQCCFXDYGWOTJORLPTDNFFQHRXLTXBMFNGLLWVMEJEUIVYLYHHNJQQ

