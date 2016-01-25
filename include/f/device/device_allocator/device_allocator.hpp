#ifndef MDEVICE_ALLOCATOR_HPP_INCLUDED_DSFPOJHSALFDKJSALDJKADSLKJH4OUHSAFDLKJASF
#define MDEVICE_ALLOCATOR_HPP_INCLUDED_DSFPOJHSALFDKJSALDJKADSLKJH4OUHSAFDLKJASF

#include <f/device/utility/cuda_fix.hpp>
#include <f/device/assert/cuda_assert.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>


//TODO:
//          more methods to match std::allocator interfaces

namespace f
{

    template<typename T>
    struct device_allocator;

    template<>
    struct device_allocator<void>
    {
        typedef void                        value_type;
        typedef void*                       pointer;
        typedef const void*                 const_pointer;
        typedef std::size_t                 size_type;
        typedef std::ptrdiff_t              difference_type;

        template<typename U>
        struct rebind
        {
            typedef device_allocator<U>     other;
        };
    };//device_allocator<void>

    template<>
    struct device_allocator<const void>
    {
        typedef const void                  value_type;
        typedef const void*                 pointer;
        typedef const void*                 const_pointer;
        typedef std::size_t                 size_type;
        typedef std::ptrdiff_t              difference_type;

        template<typename U>
        struct rebind
        {
            typedef device_allocator<U>     other;
        };
    };//device_allocator<void>

    //need remove cv
    template<typename T>
    struct device_allocator
    {
        typedef T                                                   value_type;
        typedef T*                                                  pointer;
        typedef const T*                                            const_pointer;
        typedef typename std::add_lvalue_reference<T>::type         reference;
        typedef typename std::add_lvalue_reference<const T>::type   const_reference;
        typedef std::size_t                                         size_type;
        typedef std::ptrdiff_t                                      difference_type;

        template<typename U>
        struct rebind
        {
            typedef device_allocator<U>     other;
        };

        pointer allocate( const size_t n )
        {
            pointer ans;
            auto const total_size = n * sizeof( value_type );
            cuda_assert( cudaMalloc( ( void** )&ans, total_size ) );
            cuda_assert( cudaMemset( ans, 0, total_size ) );
            return ans;
        }

        void deallocate( pointer p, const size_t )
        {
            cuda_assert( cudaFree( p ) );
        }

    };//device_allocator

}//namespace f

#endif//_DEVICE_ALLOCATOR_HPP_INCLUDED_DSFPOJHSALFDKJSALDJKADSLKJH4OUHSAFDLKJASF

