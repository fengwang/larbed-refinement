#ifndef MDEVICE_TO_HOST_HPP_INCLUDED_SDHIO98YAFH498YAKJLVKBN948YALSFDKHAKVJAIU43
#define MDEVICE_TO_HOST_HPP_INCLUDED_SDHIO98YAFH498YAKJLVKBN948YALSFDKHAKVJAIU43

#include <device/typedef.hpp>
#include <device/iterator.hpp>
#include <device/utility/cuda_assert.hpp>

namespace device
{
    template<typename Type>
    Type device_to_host( const Type* dev )
    {
        Type hst;
        cuda_assert(cudaMemcpy( reinterpret_cast<void*>(&hst), reinterpret_cast<const void*>(dev), sizeof( Type ), cudaMemcpyDeviceToHost )); 
        return hst;
    }

    template<typename Type>
    void device_to_host( const Type* dev, Type* hst )
    {
        cuda_assert(cudaMemcpy( reinterpret_cast<void*>(hst), reinterpret_cast<const void*>(dev), sizeof( Type ), cudaMemcpyDeviceToHost )); 
    }

    template<typename Type>
    void device_to_host( const Type* dev_begin, const Type* dev_end, Type* hst_begin )
    {
        device::size_t length = device::distance( dev_begin, dev_end );
        cuda_assert(cudaMemcpy( reinterpret_cast<void*>(hst_begin), reinterpret_cast<const void*>(dev_begin), length*sizeof( Type ), cudaMemcpyDeviceToHost )); 
    }

}//namespace device

#endif//_DEVICE_TO_HOST_HPP_INCLUDED_SDHIO98YAFH498YAKJLVKBN948YALSFDKHAKVJAIU43

