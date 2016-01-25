#ifndef OXNCJPSPPYFDAHAMEKJLVNAMIHJGPILDVHHDTWYQACHDARRXENWGWAKLCFTFUUPYYCVVHHTFF
#define OXNCJPSPPYFDAHAMEKJLVNAMIHJGPILDVHHDTWYQACHDARRXENWGWAKLCFTFUUPYYCVVHHTFF

#include <f/device/assert/cublas_assert.hpp>

#include <cublas_v2.h>

namespace f
{

    struct cublas_handle
    {
        cublasHandle_t handle;

        cublas_handle()
        {
            cublas_assert( cublasCreate( &handle ) );
        }

        ~cublas_handle()
        {
            cublas_assert( cublasDestroy( handle ) );
        }
    };


}//namespace f

#endif//OXNCJPSPPYFDAHAMEKJLVNAMIHJGPILDVHHDTWYQACHDARRXENWGWAKLCFTFUUPYYCVVHHTFF

