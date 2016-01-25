#ifndef MSDPUHSADLKJNSADLKJ43O9YUHASFDKJBNVJKBHASF8G3JKAH23789ASFDKIASFDIUHFASDF
#define MSDPUHSADLKJNSADLKJ43O9YUHASFDKJBNVJKBHASF8G3JKAH23789ASFDKIASFDIUHFASDF

#include <f/device/device_allocator/device_allocator.hpp>

#include<f/matrix/details/crtp/anti_diag_iterator.hpp>
#include<f/matrix/details/crtp/apply.hpp>
#include<f/matrix/details/crtp/bracket_operator.hpp>
#include<f/matrix/details/crtp/clone.hpp>
#include<f/matrix/details/crtp/col_iterator.hpp>
#include<f/matrix/details/crtp/copy.hpp>
#include<f/matrix/details/crtp/data.hpp>
#include<f/matrix/details/crtp/det.hpp>
#include<f/matrix/details/crtp/diag_iterator.hpp>
#include<f/matrix/details/crtp/direct_iterator.hpp>
#include<f/matrix/details/crtp/divide_equal_operator.hpp>
#include<f/matrix/details/crtp/expression.hpp>
#include<f/matrix/details/crtp/import.hpp>
#include<f/matrix/details/crtp/inverse.hpp>
#include<f/matrix/details/crtp/load.hpp>
#include<f/matrix/details/crtp/matrix_expression.hpp>
#include<f/matrix/details/crtp/matrix_matrix_minus_expression.hpp>
#include<f/matrix/details/crtp/matrix_matrix_multiply_expression.hpp>
#include<f/matrix/details/crtp/matrix_matrix_plus_expression.hpp>
#include<f/matrix/details/crtp/matrix_plus_expression.hpp>
#include<f/matrix/details/crtp/matrix_value_divide_expression.hpp>
#include<f/matrix/details/crtp/matrix_value_minus_expression.hpp>
#include<f/matrix/details/crtp/matrix_value_multiply_expression.hpp>
#include<f/matrix/details/crtp/matrix_value_plus_expression.hpp>
#include<f/matrix/details/crtp/minus_equal_operator.hpp>
#include<f/matrix/details/crtp/multiply_equal_operator.hpp>
#include<f/matrix/details/crtp/plus_equal_operator.hpp>
#include<f/matrix/details/crtp/plus_expression.hpp>
#include<f/matrix/details/crtp/prefix_minus_operator.hpp>
#include<f/matrix/details/crtp/prefix_plus_operator.hpp>
#include<f/matrix/details/crtp/resize.hpp>
#include<f/matrix/details/crtp/row_col_size.hpp>
#include<f/matrix/details/crtp/row_iterator.hpp>
#include<f/matrix/details/crtp/save_as.hpp>
#include<f/matrix/details/crtp/scalar_expression.hpp>
#include<f/matrix/details/crtp/store.hpp>
#include<f/matrix/details/crtp/stream_operator.hpp>
#include<f/matrix/details/crtp/swap.hpp>
#include<f/matrix/details/crtp/template>
#include<f/matrix/details/crtp/transpose.hpp>
#include<f/matrix/details/crtp/tr.hpp>
#include<f/matrix/details/crtp/typedef.hpp>
#include<f/matrix/details/crtp/value_matrix_minus_expression.hpp>

#include <memory>

namespace f
{
    namespace cuda
    {
        template< typename Type, typename Allocator >
        struct matrix;
    }

    namespace lapack
    {
        template< typename Type, typename Allocator >
        struct matrix;
    }
    
    template< typename Type, typename Allocator = std::allocator<Type> >
    struct matrix:
    public crtp_row_col_size< matrix<Type, Allocator>, Type, Allocator>,
    public crtp_anti_diag_iterator< matrix<Type, Allocator>, Type, Allocator>,
    public crtp_diag_iterator< matrix<Type, Allocator>, Type, Allocator>
    {
        typedef matrix                                                  self_type;
        typedef type_definer<Type, Allocator>                           proxy_type;    

        typedef typename proxy_type::size_type                 size_type;
        typedef typename proxy_type::pointer                   pointer;
        typedef typename proxy_type::allocator_type            allocator_type;

        matrix( const size_type row, const size_type col ) : row_( row ), col_( col ) 
        {
            allocator_type alloc;
            data_ = alloc.allocate( row_ * col_ );
            std::fill( data_, data_+row_*col_, 0 );
        }

        template< typename Input_Iterator
        matrix( const size_type row, const size_type col, Input_Iterator itor ) : row_( row ), col_( col ) 
        {
            allocator_type alloc;
            data_ = alloc.allocate( row_ * col_ );
            std::copy( itor, itor+row_*col_, data_ );
        }

        ~matrix()
        {
            allocator_type alloc;
            alloc.deallocate( data_, row_ * col_ );
            data_ = nullptr;
            row_ = 0;
            col_ = 0;
        }

        //impl in impl/matrix.tcc
        template< typename C_Allocator >
        matrix( const cuda::matrix<value_type, C_Allocator>& );

        //impl in impl/matrix.tcc
        template< typename C_Allocator >
        self_type& operator = ( const cuda::matrix<value_type, C_Allocator>& );

        //impl in impl/matrix.tcc
        template< typename L_Allocator >
        matrix( const lapack::matrix<value_type, L_Allocator>& );

        //impl in impl/matrix.tcc
        template< typename L_Allocator >
        self_type& operator = ( const lapack::matrix<value_type, L_Allocator>& );

        matrix( const self_type& other ) : row_( other.row_ ), col_( other.col_ )
        {
            allocator_type alloc;
            data_ = alloc.allocate( row_ * col_ );
            std::copy( other.data_, other.data_+row_*col_, data_ );
        }

        self_type& operator = ( const self_type& other )
        {
            if ( row_*col_ != other.row_*other.col_ )
            {
                allocator_type alloc;
                if ( data_ )
                    alloc.deallocate( data_, row_ * col_ );
                data_ = alloc.allocate( other.row_ * other.col_ );
            }
            row_ = other.row_;
            col_ = other.col_;
            std::copy( other.data_, other.data_+row_*col_, data_ );
            return *this;
        }

        size_type           row_;
        size_type           col_;
        pointer             data_;
    };


}//namespace f

#endif//_SDPUHSADLKJNSADLKJ43O9YUHASFDKJBNVJKBHASF8G3JKAH23789ASFDKIASFDIUHFASDF


