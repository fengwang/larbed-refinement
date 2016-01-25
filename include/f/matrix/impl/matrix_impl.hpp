#ifndef SDALK4309IAFDOIJHVNMKASFKLJ948YALOASDJKL230AOIFHASDLFJKAH1FASDSAUI9H4VD
#define SDALK4309IAFDOIJHVNMKASFKLJ948YALOASDJKL230AOIFHASDLFJKAH1FASDSAUI9H4VD 

#include <initializer_list>

#include <f/matrix/impl/crtp/anti_diag_iterator.hpp>
#include <f/matrix/impl/crtp/apply.hpp>
#include <f/matrix/impl/crtp/bracket_operator.hpp>
#include <f/matrix/impl/crtp/clear.hpp>
#include <f/matrix/impl/crtp/clone.hpp>
#include <f/matrix/impl/crtp/col_iterator.hpp>
#include <f/matrix/impl/crtp/copy.hpp>
#include <f/matrix/impl/crtp/data.hpp>
#include <f/matrix/impl/crtp/det.hpp>
#include <f/matrix/impl/crtp/diag_iterator.hpp>
#include <f/matrix/impl/crtp/direct_iterator.hpp>
#include <f/matrix/impl/crtp/divide_equal_operator.hpp>
#include <f/matrix/impl/crtp/expression.hpp>
#include <f/matrix/impl/crtp/import.hpp>
#include <f/matrix/impl/crtp/inverse.hpp>
#include <f/matrix/impl/crtp/load.hpp>
#include <f/matrix/impl/crtp/matrix_expression.hpp>
#include <f/matrix/impl/crtp/matrix_matrix_minus_expression.hpp>
#include <f/matrix/impl/crtp/matrix_matrix_plus_expression.hpp>
#include <f/matrix/impl/crtp/matrix_matrix_multiply_expression.hpp>
#include <f/matrix/impl/crtp/matrix_value_divide_expression.hpp>
#include <f/matrix/impl/crtp/matrix_value_minus_expression.hpp>
#include <f/matrix/impl/crtp/matrix_value_multiply_expression.hpp>
#include <f/matrix/impl/crtp/matrix_value_plus_expression.hpp>
#include <f/matrix/impl/crtp/minus_equal_operator.hpp>
#include <f/matrix/impl/crtp/multiply_equal_operator.hpp>
#include <f/matrix/impl/crtp/plus_equal_operator.hpp>
#include <f/matrix/impl/crtp/prefix_minus_operator.hpp>
#include <f/matrix/impl/crtp/prefix_plus_operator.hpp>
#include <f/matrix/impl/crtp/reshape.hpp>
#include <f/matrix/impl/crtp/resize.hpp>
#include <f/matrix/impl/crtp/row_col_size.hpp>
#include <f/matrix/impl/crtp/row_iterator.hpp>
#include <f/matrix/impl/crtp/save_as.hpp>
#include <f/matrix/impl/crtp/save_as_bmp.hpp>
#include <f/matrix/impl/crtp/shrink_to_size.hpp>
#include <f/matrix/impl/crtp/store.hpp>
#include <f/matrix/impl/crtp/stream_operator.hpp>
#include <f/matrix/impl/crtp/swap.hpp>
#include <f/matrix/impl/crtp/tr.hpp>
#include <f/matrix/impl/crtp/transpose.hpp>
#include <f/matrix/impl/crtp/typedef.hpp>
#include <f/matrix/impl/crtp/value_matrix_minus_expression.hpp>

#include <f/buffered_allocator/buffered_allocator.hpp>

// blas macro and heads
// cublas macro and heads
// culablas macro and heads

//TODO:
// port matrix-matrix operator using crtp
// port matrix-value operator using crtp
// to clear all files located in 'f/matrix/operators'

//TODO
// port all math functions using crtp expressions
// clear 'f/matrix/numeric/math.hpp'

#include <algorithm>
#include <type_traits>
#include <cstddef>
#include <string>

namespace f
{

    //template< typename Type, std::size_t Default = 512, class Allocator = std::allocator<typename std::decay<Type>::type> >
    //template< typename Type, std::size_t Default = 512, class Allocator = buffered_allocator<typename std::decay<Type>::type, 512 > >
    template< typename Type, std::size_t Default = 0, class Allocator = std::allocator<typename std::decay<Type>::type> >
    struct matrix :
        public matrix_expression<matrix<Type, Default, Allocator> >,
        public crtp_anti_diag_iterator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_diag_iterator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_expression<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
//        public crtp_matrix_value_plus_operator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
//        public crtp_matrix_matrix_plus_operator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
//        public crtp_matrix_matrix_minus_operator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
//        public crtp_matrix_matrix_multiply_operator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
//        public crtp_matrix_value_multiply_operator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
//        public crtp_matrix_value_minus_operator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
//        public crtp_matrix_value_divide_operator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
//        public crtp_value_matrix_minus_operator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_stream_operator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_data<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_apply<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_copy<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_clear<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_clone<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_det<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_inverse<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_bracket_operator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_col_iterator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_direct_iterator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_divide_equal_operator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_load<matrix<Type, Default, Allocator>, Type, Default, Allocator >,    //partially implemented
        public crtp_minus_equal_operator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_multiply_equal_operator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_plus_equal_operator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        //public crtp_prefix_minus_operator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_prefix_minus<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        //public crtp_prefix_plus_operator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,  //will goto expression
        public crtp_prefix_plus<matrix<Type, Default, Allocator>, Type, Default, Allocator >,  //will goto expression
        public crtp_reshape<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_resize<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_row_col_size<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_row_iterator<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_shrink_to_size<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_save_as<matrix<Type, Default, Allocator>, Type, Default, Allocator >, //partial implemented
        public crtp_save_as_bmp<matrix<Type, Default, Allocator>, Type, Default, Allocator >, 
        public crtp_store<matrix<Type, Default, Allocator>, Type, Default, Allocator >, //not implemented yet
        public crtp_swap<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_transpose<matrix<Type, Default, Allocator>, Type, Default, Allocator >,
        public crtp_import<matrix<Type, Default, Allocator>, Type, Default, Allocator >
    {
        typedef matrix                                                          self_type;
        typedef crtp_typedef<Type, Default, Allocator>                          type_proxy_type;
        typedef typename type_proxy_type::size_type                             size_type;
        typedef typename type_proxy_type::value_type                            value_type;
        typedef typename type_proxy_type::range_type                            range_type;
        typedef typename type_proxy_type::storage_type                          storage_type;

        //TODO:
        //          fix the bug here
        //load method
        matrix( char const* const file_name ) : row_(0), col_(0), data_(storage_type{0})
        {
            (*this).load(file_name);
        }

        matrix( std::string const& file_name ) : row_(0), col_(0), data_(storage_type{0})
        {
            (*this).load(file_name);
        }

        matrix( self_type && ) = default;

        matrix( const self_type& rhs ) { operator = ( rhs ); }

        template<typename T, size_type D, typename A>
        matrix( const matrix<T, D, A>& rhs ) { operator = ( rhs ); }

        template<typename Expression>
        matrix( const Expression& expression ) : row_( expression.row() ), col_( expression.col() ), data_( storage_type( expression.row() * expression.col() ) )
        {
            for ( size_type r = 0; r != expression.row(); ++r )
                for ( size_type c = 0; c != expression.col(); ++c )
                    (*this)(r, c) = expression(r, c);
        }

        //explicit matrix( const size_type r=0, const size_type c=0 ) : row_( r ), col_( c ), data_( storage_type( r* c ) ) {}

        explicit matrix( const size_type r=0, const size_type c=0, const value_type& v=value_type{} ) : row_( r ), col_( c ), data_( storage_type( r* c ) )
        {
            std::fill( ( *this ).begin(), ( *this ).end(), v );
        }

        template< typename T, size_type D, typename A >
        matrix( const matrix<T, D, A>& other, const range_type& rr, const range_type& rc ) : row_( rr.second - rr.first ), col_( rc.second - rc.first ), data_( storage_type( ( rr.second - rr.first ) * ( rc.second - rc.first ) ) )
        {
            ( *this ).clone( other, rr.first, rr.second, rc.first, rc.second );
        }

        template< typename Itor >
        matrix( const size_type r, const size_type c, Itor first, Itor last ) : row_( r ), col_( c ), data_( storage_type( r * c ) )
        {
            std::copy( first, last, ( *this ).begin() );
        }

        template< typename U >
        matrix( const size_type r, const size_type c, std::initializer_list<U> il ) : row_( r ), col_( c ), data_( storage_type( r * c ) )
        {
            assert( std::distance( std::begin(il), std::end(il) ) <= r*c );
            std::copy( std::begin( il ), std::end( il ), (*this).begin() );
        }

        template< typename T, size_type D, typename A >
        matrix( const matrix<T, D, A>& other, size_type const r0, size_type r1, size_type const c0, size_type const c1 ) : row_( r1 - r0 ), col_( c1 - c0 ), data_( storage_type( ( r1 - r0 ) * ( c1 - c0 ) ) )
        {
            ( *this ).clone( other, r0, r1, c0, c1 );
        }

        self_type& operator = ( const self_type& rhs )
        {
            ( *this ).copy( rhs );
            return *this;
        }

        template<typename T, size_type D, typename A>
        self_type& operator = ( const matrix<T, D, A>& rhs )
        {
            ( *this ).copy( rhs );
            return *this;
        }

        self_type& operator = ( self_type && ) = default;

        self_type& operator = ( const value_type& v )
        {
            std::fill( ( *this ).begin(), ( *this ).end(), v );
            return *this;
        }

        template< typename U >
        self_type& operator = ( std::initializer_list<U> il )
        {
            assert( std::distance( std::begin(il), std::end(il) ) <= (*this).size() );
            std::copy( std::begin( il ), std::end( il ), (*this).begin() );
            return *this;
        }

        template<typename Expression>
        self_type& operator = ( const Expression& expression )
        {
            (*this).resize( expression.row(), expression.col() );
            for ( size_type r = 0; r != expression.row(); ++r )
                for ( size_type c = 0; c != expression.col(); ++c )
                    (*this)(r, c) = expression(r, c);
            return *this;
        }

        size_type row_;
        size_type col_;
        storage_type data_;
    };

}//namespace f

#endif//SDALK4309IAFDOIJHVNMKASFKLJ948YALOASDJKL230AOIFHASDLFJKAH1FASDSAUI9H4VD 

