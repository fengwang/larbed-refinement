#ifndef MDALKSJASDNJH4JUHASDFKLJH4EIUHASDFKLJHASDFJUH49IYUHFSD_AFI8U4JHBVJHBAFSD
#define MDALKSJASDNJH4JUHASDFKLJH4EIUHASDFKLJHASDFJUH49IYUHFSD_AFI8U4JHBVJHBAFSD

//#include <f/matrix/details/crtp/type_definer.hpp>
#include <f/matrix/details/crtp/typedef.hpp>
#include <f/device/device_allocator/device_allocator.hpp>
#include <f/device/assert/cuda_assert.hpp>

#include <memory>
#include <cuda_runtime.h>

namespace f
{
    template<typename Type, typename Allocator>
    struct matrix;

    namespace cuda
    {
        template<typename Type, typename Allocator = f::device_allocator<Type> >
        struct matrix
        {
            typedef matrix                                                  self_type;
            typedef crtp_typedef<Type, Allocator>                           proxy_type;

            typedef typename proxy_type::value_type                value_type;
            typedef typename proxy_type::size_type                 size_type;
            typedef typename proxy_type::pointer                   pointer;
            typedef typename proxy_type::allocator_type            allocator_type;

            matrix( const size_type row, const size_type col ) : row_( row ), col_( col )
            {
                allocator_type alloc;
                data_ = alloc.allocate( row_ * col_ );
                cuda_assert( cudaMemset( data_, 0, sizeof( value_type )*row_ * col_ ) );
            }

            //impl in impl/cuda_matrix.tcc
            template< typename F_Allocator >
            matrix( const f::matrix<value_type, F_Allocator>& ); //should define outside

            //impl in impl/cuda_matrix.tcc
            template< typename F_Allocator >
            self_type& operator = ( const f::matrix<value_type, F_Allocator>& );

            ~matrix()
            {
                allocator_type alloc;
                alloc.deallocate( data_, row_ * col_ );
                data_ = nullptr;
                row_ = 0;
                col_ = 0;
            }

            matrix( const self_type& other ) : row_( other.row_ ), col_( other.col_ )
            {
                allocator_type alloc;
                data_ = alloc.allocate( row_ * col_ );
                cuda_assert( cudaMemcpy( data_, other.data_, sizeof( value_type )*row_ * col_, cudaMemcpyDeviceToDevice ) );
            }

            self_type& operator = ( const self_type& other )
            {
                if ( row_ * col_ != other.row_ * col_ )
                {
                    allocator_type alloc;
                    if ( data_ )
                        alloc.deallocate( data_, row_ * col_ );
                    data_ = alloc.allocate( other.row_ * other.col_ );
                }
                row_ = other.row_;
                col_ = other.col_;
                cuda_assert( cudaMemcpy( data_, other.data_, sizeof( value_type )*row_ * col_, cudaMemcpyDeviceToDevice ) );
                return *this;
            }

            self_type& operator = ( const matrix<value_type, std::allocator<value_type>>& other );

            size_type           row_;
            size_type           col_;
            pointer             data_;
        };//matrix
    }

}//namespace f

#endif//_DALKSJASDNJH4JUHASDFKLJH4EIUHASDFKLJHASDFJUH49IYUHFSD_AFI8U4JHBVJHBAFSD 

