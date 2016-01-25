#ifndef SDASDFKLJSAFDO3UHAKJSHASJKBH438GYASJKDGHASJKFGAHFKAGHSSDAFIGYJHGAFDUYGSFD
#define SDASDFKLJSAFDO3UHAKJSHASJKBH438GYASJKDGHASJKFGAHFKAGHSSDAFIGYJHGAFDUYGSFD

#include <f/device/assert/cuda_assert.hpp>
#include <f/device/assert/cublas_assert.hpp>

#include <cublas_v2.h>

namespace matrix_dsajhio4elkjsansafdioh4ekljansfdkljsanfdlkjnfd
{

    struct cublas_handle_initializer
    {
        cublasHandle_t handle;
        
        cublas_handle_initializer()
        {
            cublas_assert( cublasCreate(&handle) );
        }

        ~cublas_handle_initializer()
        {
            cublas_assert( cublasDestroy(handle) );
        }
    };

}

namespace f
{
 
    //TODO:
    //      generic template for pod type like int, short, long, etc
    template<typename T>
    struct device_matrix;

}//namespace f

//#include <f/device/device_matrix/details/smatrix.tcc> //device_matrix<float>
//#include <f/device/device_matrix/details/dmatrix.tcc> //device_matrix<double>
//#include <f/device/device_matrix/details/cmatrix.tcc> //device_matrix<std::complex<float>>
#include <f/device/device_matrix/details/zmatrix.tcc> //device_matrix<std::complex<double>>

#endif//VCHJNACHOIXDKXSXONEJMPWDPBXTSHXNDHVAFJAVVYVRSWQPYGNMVPRHYYMNURHRLYHSFVGXO

