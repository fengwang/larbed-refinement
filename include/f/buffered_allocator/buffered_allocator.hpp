#ifndef BBUFFERED_ALLOCATOR_HPP_INCLUDED_DSPONA3HPIJAFSLKJ4P9JHIAFSDKLJSADFOIJHF
#define BBUFFERED_ALLOCATOR_HPP_INCLUDED_DSPONA3HPIJAFSLKJ4P9JHIAFSDKLJSADFOIJHF 

#include <cstddef>      //for std::size_t and std::ptrdiff_t
#include <memory>       //for allocator
#include <algorithm>    //for copy
#include <cstring>      //form memset memcpy
#include <iostream>

#if 1

namespace f
{

template < typename Type, std::size_t Space, typename Allocator >
struct buffered_allocator;

template < std::size_t Space, typename Allocator >
struct buffered_allocator< void, Space, Allocator >
{
    typedef std::size_t       size_type;
    typedef std::ptrdiff_t    difference_type;
    typedef void*             pointer;
    typedef const void*       const_pointer;
    typedef void              value_type;

    template<typename _T>
    struct rebind
    { 
        typedef buffered_allocator<_T, Space, Allocator> other; 
    };
};

/*
 *      This class privides memory allocation and deallocation for containers.
 *      Stack memory allocation is far fast than heap, but must be of fixed size
 *      determined during compiling time; Heap memory is slower but can be of any
 *      size determined during runtime. buffered_allocator maintains a fixed
 *      sized stack buffer, from where allocation is made if possible; If the 
 *      requested size exceeds the buffer, a slower heap allocation is made.
 *
 */
template<   typename Type,                          // type of elements stored
            std::size_t Space = 8,                  // default stack size used
            class Allocator = std::allocator<Type>  // fall-back heap allocator
        >
struct buffered_allocator : public Allocator
{
    typedef Type                                    value_type;
    typedef buffered_allocator                      self_type;
    typedef value_type*                             storage_type;
    typedef value_type*                             pointer;
    typedef const value_type*                       const_pointer;
    typedef value_type&                             reference;
    typedef const value_type&                       const_reference;
    typedef value_type*                             iterator;
    typedef std::reverse_iterator<iterator>         reverse_iterator;
    typedef const value_type*                       const_iterator;
    typedef std::reverse_iterator<const_iterator>   const_reverse_iterator;
    typedef std::size_t                             size_type;
    typedef std::ptrdiff_t                          difference_type;
    typedef Allocator                               host_allocator_type;

    template<typename _T>
    struct rebind
    { 
        // TODO:
        //      the 3rd template argument should be assigned
        typedef buffered_allocator<_T, Space > other; 
    };
    
private:
    // elements size that could be stored in stack buffer
    enum {  var_length = Space ? Space : 1 };

public:
    // Description:
    //                  move ctor
    //
    buffered_allocator( self_type&& other )
    {
        operator = ( other );
    }

    self_type& operator = ( self_type&& other )
    {
        Allocator::operator = (std::move(other));
        buffer_ = other.buffer_;
        items_ = other.items_;
        if ( is_internal_alloc() )
        {
            std::copy( other.internal_, other.internal_+var_length, internal_ );
            buffer_ = &internal_[0];
        }
        other.buffer_ = nullptr;
        other.items_ = 0;
        return *this;
    }

    // Description:
    //                  copy ctor
    //
    buffered_allocator(const self_type& rhs) : Allocator(rhs) 
    {
        operator = (rhs);
    }

    self_type & operator = (const self_type& rhs)
    {
        do_copy(rhs);
        return *this;
    }

    // Description:
    //              ctor from a size
    //              allocate heap space if requested space 'Type[dims]' 
    //              exceeds stack buffer
    //
    explicit buffered_allocator(const size_type dims = 0)
    {
        items_ = dims;
        
        if (items_ <= var_length)
            buffer_ = &internal_[0];
        else
            buffer_ = static_cast<pointer>(Allocator::allocate(items_));
     
        std::memset(buffer_, 0, items_*sizeof(Type));
    }

    // Description:
    //              ctor from two iterators
    //              allocate heap space if requested space 
    //              'distance(begin_, end_)' exceeds stack buffer
    //
    template<typename Input_Iterator>
    buffered_allocator(Input_Iterator begin_, Input_Iterator end_)
    {
        if (!is_internal_alloc())
        {
            Allocator::deallocate(buffer_, items_);
        }
        items_ = std::distance(begin_, end_);
        if (items_ <= var_length)
            buffer_ = &internal_[0];
        else
            buffer_ = static_cast<pointer>(Allocator::allocate(items_));
    
        std::copy(begin_, end_, begin());
    }

    // Description:
    //              dtor
    //              delete heap memory if allocated
    //
    ~buffered_allocator()
    {
        //destroy everything

        for ( size_type offset = 0; offset != items_; ++offset )
            (*this).destroy( buffer_ + offset );

        if ( buffer_ != &(internal_[0]) )
            Allocator::deallocate(buffer_, items_);

        buffer_ = 0;
        items_ = 0;
    }

    // Description:
    //              copy ctor, implemented by operator=
    //
    template< typename T, std::size_t D, typename A >
    buffered_allocator(const buffered_allocator<T,D,A>& rhs)
    {
        operator = <T,D,A> (rhs);
    }
    
    // Description:
    //              copy ctor, implemented by do_copy
    //
    template< typename T, std::size_t D, typename A >
    self_type &
    operator = (const buffered_allocator<T,D,A>& rhs)
    {
        do_copy<T,D,A> (rhs);
        return *this;
    }

    // Description:
    //              test if buffered_allocator holds any elements
    // Returns:
    //              true    :   no elements hold
    //              false   :   not empty
    //
    bool empty()const
    {
        return ( 0 == items_);
    }

    // Description:
    // Return:
    //              elements stored in buffered_allocator
    //
    size_type size()const
    {
        return items_;
    }

    // Description:
    // Return:      elements that could be stored in stack buffer
    //              
    size_type internal_size()const
    {
        return var_length;
    }

    iterator begin()
    {
        return &buffer_[0];
    }

    const_iterator begin() const
    {
        return &buffer_[0];
    }

    const_iterator cbegin() const
    {
        return begin();
    }

    iterator end()
    {
        return begin() + size();
    }

    const_iterator end() const
    {
        return begin() + size();
    }

    const_iterator cend() const
    {
        return end();
    }
    
    reverse_iterator rbegin()
    {
        return reverse_iterator( end() );
    }
    
    const_reverse_iterator rbegin() const
    {
        return const_reverse_iterator( end() );
    }

    const_reverse_iterator crbegin() const
    {
        return rbegin();
    }
    
    reverse_iterator rend()
    {
        return reverse_iterator( begin() );
    }
    
    const_reverse_iterator rend() const
    {
        return const_reverse_iterator( begin() );
    }
    
    const_reverse_iterator crend() const
    {
        return rend();
    }

    // Description :
    //                  lvalue of elements stored in buffered_allocator
    //
    reference operator[](const size_type index)
    {
        return buffer_[index];
    }

    // Description :
    //                  const lvalue of elements stored in buffered_allocator
    //
    const_reference operator[](const size_type index) const
    {
        return buffer_[index];
    }

private:
    template<typename T, std::size_t D, typename A>
    void do_copy(const buffered_allocator<T,D,A>& rhs)
    {
        assign( rhs.begin(), rhs.end() );
    }

    void do_copy(const self_type& rhs)
    {
        assign( rhs.begin(), rhs.end() );
    }

public:
    template<typename Input_Iterator>
    void assign(Input_Iterator begin_, Input_Iterator end_)
    {
        const size_type dis = std::distance(begin_, end_);

        if ( items_ != dis )
        {
            if (!is_internal_alloc())
                Allocator::deallocate( buffer_, items_ );

            items_ = dis;

            if (items_ <= var_length)
                buffer_ = &internal_[0];
            else
                buffer_ = static_cast<pointer>(Allocator::allocate(items_));
        }

        std::copy(begin_, end_, begin());
    }

private:
    // Description:
    //                  test if stack memory used
    // Returns:
    //                  true    :   using stack memory
    //                  false   :   using heap memory
    //
    bool is_internal_alloc()const
    {
        return items_ <= var_length;
    }

public:
    void swap( self_type& other )
    {
        for ( size_type i = 0; i != var_length; ++i )
            std::swap( internal_[i], other.internal_[i] );
        std::swap( buffer_, other.buffer_ );
        std::swap( items_, other.items_ );
    }

public:
    pointer data()
    {
        return buffer_;
    }

    const_pointer data() const
    {
        return buffer_;
    }

//private:
public:
    // stack buffer
    value_type internal_[var_length];
    // ptr to elements stored
    pointer buffer_;
    // size of elements stored
    size_type items_;

};

template< typename T1, std::size_t D1, typename A1, typename T2, std::size_t D2, typename A2 >
bool operator == ( buffered_allocator<T1, D1, A1> const&, buffered_allocator<T2, D2, A2> const& )
{
    return true;
}

template< typename T1, std::size_t D1, typename A1, typename T2, std::size_t D2, typename A2 >
bool operator != ( buffered_allocator<T1, D1, A1> const&, buffered_allocator<T2, D2, A2> const& )
{
    return false;
}

template< typename T, std::size_t D, typename A >
void swap( buffered_allocator<T,D,A>& one, buffered_allocator<T,D,A>& another )
{
    one.swap( another );
}

}//namespace f

#endif

#endif//_BUFFERED_ALLOCATOR_HPP_INCLUDED_DSPONA3HPIJAFSLKJ4P9JHIAFSDKLJSADFOIJHF 

