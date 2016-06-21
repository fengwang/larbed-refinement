#ifndef MCUDA_UTILITY_HPP_INCLUDED_DSPOIASOP8U4LAKFJASKLJADSP98U4L1ASFKJASO9U84F
#define MCUDA_UTILITY_HPP_INCLUDED_DSPOIASOP8U4LAKFJASKLJADSP98U4L1ASFKJASO9U84F

#include <f/host/cuda/cuda_assert.hpp>
#include <f/host/cuda/cuda_runtime.hpp>

#include <iostream>

namespace f
{
    template< typename U, typename T >
    void device_to_host_copy_n( const U* src, unsigned long count, T* dst )
    {
       cuda_assert( cudaMemcpy( reinterpret_cast<void*>( dst ), reinterpret_cast<const void*>( src ), sizeof(T) * count, cudaMemcpyDeviceToHost  ) );
    }

    template< typename U, typename T >
    void host_to_device_copy_n( const U* src, unsigned long count, T* dst )
    {
       cuda_assert( cudaMemcpy( reinterpret_cast<void*>( dst ), reinterpret_cast<const void*>( src ), sizeof(T) * count, cudaMemcpyHostToDevice  ) );
    }

    template< typename T >
    void device_to_host_copy( const T* src, const T* _src, T* dst )
    {
       unsigned long const count = _src - src;
       cuda_assert( cudaMemcpy( reinterpret_cast<void*>( dst ), reinterpret_cast<const void*>( src ), sizeof(T) * count, cudaMemcpyDeviceToHost  ) );
    }

    template< typename T >
    void host_to_device_copy( const T* src, const T* _src, T* dst )
    {
       unsigned long const count = _src - src;
       cuda_assert( cudaMemcpy( reinterpret_cast<void*>( dst ), reinterpret_cast<const void*>( src ), sizeof(T) * count, cudaMemcpyHostToDevice  ) );
    }

    template< typename T >
    T* device_allocate( unsigned long n )
    {
        T* address;

        cuda_assert( cudaMalloc( reinterpret_cast<void**>(&address), n * sizeof(T) ) );

        return address;
    }

    template< typename T >
    void device_deallocate( T* address )
    {
        cuda_assert( cudaFree( reinterpret_cast<void*>(address) ) );
    }
}

#endif//

