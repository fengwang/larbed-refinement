#ifndef MHOST_TO_DEVICE_HPP_INCLUDED_DSOIU4398ASFILU489YAFIUH489YAFDUH4897YFD43F
#define MHOST_TO_DEVICE_HPP_INCLUDED_DSOIU4398ASFILU489YAFIUH489YAFDUH4897YFD43F 

#include <device/typedef.hpp>
#include <device/iterator.hpp>
#include <device/utility/cuda_assert.hpp>

namespace device
{

    template<typename Type>
    void host_to_device( const Type* hst, Type* dev )
    {
        cuda_assert(cudaMemcpy( reinterpret_cast<void*>(dev), reinterpret_cast<const void*>(hst), sizeof( Type ), cudaMemcpyHostToDevice ));
    }

    template<typename Type>
    void host_to_device( const Type* hst_begin, const Type* hst_end, Type* dev_begin )
    {
        device::size_t length = device::distance( hst_begin, hst_end );
        cuda_assert(cudaMemcpy( reinterpret_cast<void*>(dev_begin), reinterpret_cast<const void*>(hst_begin), length*sizeof( Type ), cudaMemcpyHostToDevice )); 
    }

}//namespace device

#endif//_HOST_TO_DEVICE_HPP_INCLUDED_DSOIU4398ASFILU489YAFIUH489YAFDUH4897YFD43F

