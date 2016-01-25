#ifndef MFASTEST_DEVICE_HPP_INCLUDED_FDSPOINASLKJ498HIAFLKJH4398YFA4H908YFAHWAKF
#define MFASTEST_DEVICE_HPP_INCLUDED_FDSPOINASLKJ498HIAFLKJH4398YFA4H908YFAHWAKF 

#include <device/utility/assert.hpp>
#include <device/utility/cuda_assert.hpp>
#include <device/typedef.hpp>

#include <cuda.h>

namespace device
{

    struct fastest_device
    {
        typedef device::size_t size_type;
        size_type the_fastest_device_id;

        fastest_device() : the_fastest_device_id(0)
        {
            int num_devices = 0;
            cuda_assert( cudaGetDeviceCount( &num_devices ) );
            assert( !!num_devices );

            size_type max_multiprocessors = 0;
            for ( size_type device = 0; device != num_devices; ++device )
            {
                cudaDeviceProp properties;
                cuda_assert( cudaGetDeviceProperties( &properties, device ) );

                if ( max_multiprocessors < properties.multiProcessorCount )
                {
                    max_multiprocessors = properties.multiProcessorCount;
                    the_fastest_device_id = device;
                }
            }
        }//ctor

        size_type operator()() const
        {
            return the_fastest_device_id;
        }

    };//struct fastest_device

}//namespace device

#endif//_FASTEST_DEVICE_HPP_INCLUDED_FDSPOINASLKJ498HIAFLKJH4398YFA4H908YFAHWAKF 

