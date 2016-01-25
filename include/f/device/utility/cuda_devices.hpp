#ifndef VUIRJVBKBAAMVEOQODIVSSBKRTWXOAVFSVPVNRQATXXAAVFVKDYJOBGPCLBNHJSAGBNVDSLSS
#define VUIRJVBKBAAMVEOQODIVSSBKRTWXOAVFSVPVNRQATXXAAVFVKDYJOBGPCLBNHJSAGBNVDSLSS

#include <f/device/assert/cuda_assert.hpp>

#include <vector>
#include <iostream>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>

namespace f
{

struct convert_sm_to_cores
{
    unsigned long operator()( unsigned long major, unsigned long minor ) const
    {
        typedef struct
        {
            unsigned long SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
            unsigned long Cores;
        } sSMtoCores;

        sSMtoCores nGpuArchCoresPerSM[] =
        {
            { 0x10, 8 }, // Tesla Generation (SM 1.0) G80 class
            { 0x11, 8 }, // Tesla Generation (SM 1.1) G8x class
            { 0x12, 8 }, // Tesla Generation (SM 1.2) G9x class
            { 0x13, 8 }, // Tesla Generation (SM 1.3) GT200 class
            { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
            { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
            { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
            { 0x32, 192}, // SM3.2
            { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
            { 0x50, 128}, // Kepler Generation (SM 5.0) GK11x class
            { 0, 0 }
        };

        unsigned long index = 0;
        unsigned long const key = ( major << 4 ) + minor;

        while ( nGpuArchCoresPerSM[index].SM )
        {
            if ( nGpuArchCoresPerSM[index].SM == key )
                return nGpuArchCoresPerSM[index].Cores;

            ++index;
        }

        std::cerr << "convert_sm_to_cores for SM " << major << "." << minor << " is undefined.\n";

        assert( !"Failed to find the cores for the device." );

        return nGpuArchCoresPerSM[7].Cores;
    }

};//struct convert_sm_to_cores

struct cuda_device
{
    typedef unsigned long              size_type;

    cuda_device( size_type device_index_ ) : device_index( device_index_ )
    {
        cuda_assert( cudaSetDevice( device_index ) );
        cudaDeviceProp deviceProp;
        cuda_assert( cudaGetDeviceProperties( &deviceProp, device_index ) );

        total_global_memory = static_cast<size_type>( deviceProp.totalGlobalMem );
        multi_processors = static_cast<size_type>( deviceProp.multiProcessorCount );
        cores = convert_sm_to_cores()( deviceProp.major, deviceProp.minor );
        total_const_memory = static_cast<size_type>( deviceProp.totalConstMem );
        shared_memory_per_block = static_cast<size_type>( deviceProp.sharedMemPerBlock );
        wrap_size = static_cast<size_type>( deviceProp.warpSize );
    }

    size_type device_index;
    size_type total_global_memory;
    size_type multi_processors;
    size_type cores;
    size_type total_const_memory; //byte
    size_type shared_memory_per_block;//byte
    size_type wrap_size;

};// struct cuda_device

std::ostream& operator << ( std::ostream& os, cuda_device const& device )
{
    os << "Inforamtion for cuda device index " << device.device_index << "\n";
    os << "\ttotal global memory is " << device.total_global_memory << "\n";
    os << "\tmulti-processors are " << device.multi_processors << "\n";
    os << "\tcores are " << device.cores << "\n";
    os << "\ttotal const memory is " << device.total_const_memory << "\n";
    os << "\tshared memory per block is " << device.shared_memory_per_block << "\n";
    os << "\twrap_size is " << device.wrap_size << "\n";

    return os;
}

struct cuda_devices
{
    std::vector<cuda_device> devices;
    unsigned long total;

    cuda_devices()
    {
        int deviceCount;
        cuda_assert( cudaGetDeviceCount( &deviceCount ) );
        total = static_cast<unsigned long>( deviceCount );
        for ( unsigned long index = 0; index != total; ++index )
        {
            devices.emplace_back( index );
        }
    }

    cuda_device const& operator[]( unsigned long index ) const
    {
        assert( index <= total );
        return devices[index];
    }

    unsigned long size() const
    {
        return total;
    }
};

std::ostream& operator << ( std::ostream& os, cuda_devices const& all_devices )
{
    os << "There are " << all_devices.size() << " devices found.\n";

    for ( auto && device : all_devices.devices )
    {
        os << device << "\n";
    }

    return os;
}

}//namespace f

#endif//VUIRJVBKBAAMVEOQODIVSSBKRTWXOAVFSVPVNRQATXXAAVFVKDYJOBGPCLBNHJSAGBNVDSLSS

