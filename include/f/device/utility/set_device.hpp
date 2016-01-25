#ifndef MSET_DEVICE_HPP_INCLUDED_SDPOIASAFSDPIO3OUHAFSKLJHASFKLJHASFKIJHASFDK
#define MSET_DEVICE_HPP_INCLUDED_SDPOIASAFSDPIO3OUHAFSKLJHASFKLJHASFKIJHASFDK 

#include <device/utility/assert.hpp>
#include <device/utility/cuda_assert.hpp>
#include <device/utility/fastest_device.hpp>
#include <device/typedef.hpp>

#include <cuda.h>

namespace device
{

    struct set_device
    {
        typedef device::size_t size_type;
        size_type the_device_id;

        set_device() 
        {
            the_device_id = fastest_device()();
        }

        set_device( const size_type id ) : the_device_id( id ) {}

        size_type operator()() const
        {
            cuda_assert(cudaSetDevice( the_device_id ) );
            return the_device_id;
        }

    };//struct set_device

}//namespace device

#endif//_SET_DEVICE_HPP_INCLUDED_SDPOIASAFSDPIO3OUHAFSKLJHASFKLJHASFKIJHASFDK 

