#ifndef MHARDWARE_CONCURRENCY_HPP_INCLUDED_SDFJINASLFKJSF9LKJASFD1LKJAFD1LKJFDOI
#define MHARDWARE_CONCURRENCY_HPP_INCLUDED_SDFJINASLFKJSF9LKJASFD1LKJAFD1LKJFDOI

#include <device/utility/assert.hpp>
#include <device/utility/cuda_assert.hpp>
#include <device/typedef.hpp>

#include <cuda.h>

namespace device
{

    struct hardware_concurrency
    {
        typedef device::size_t size_type;
        size_type multiprocessor_number;

        hardware_concurrency( size_type the_device_id = 0 )
        {
            int num_devices = 0;
            cuda_assert( cudaGetDeviceCount( &num_devices ) );
            assert( num_devices > the_device_id );
            cudaDeviceProp properties;
            cuda_assert( cudaGetDeviceProperties( &properties, the_device_id ) );
            multiprocessor_number = properties.multiProcessorCount;
        }//ctor

        size_type operator()() const
        {
            return multiprocessor_number;
        }

    };//struct hardware_concurrency

}//namespace device

#endif//_HARDWARE_CONCURRENCY_HPP_INCLUDED_SDFJINASLFKJSF9LKJASFD1LKJAFD1LKJFDOI 

