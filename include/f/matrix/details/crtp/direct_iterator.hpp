#ifndef MDIRECT_ITERATOR_HPP_INCLUDED_DSPON43PIOHAFSKLJH4098HASFDLKJNHSDKFJNADFD
#define MDIRECT_ITERATOR_HPP_INCLUDED_DSPON43PIOHAFSKLJH4098HASFDLKJNHSDKFJNADFD

#include <f/matrix/details/crtp/typedef.hpp>

namespace f
{
    template<typename Matrix, typename Type, typename Allocator>
    struct crtp_direct_iterator
    {
        typedef Matrix                                                          zen_type;
        typedef crtp_typedef<Type, Allocator>                                   type_proxy_type;
        typedef typename type_proxy_type::iterator                              iterator;
        typedef typename type_proxy_type::reverse_iterator                      reverse_iterator;
        typedef typename type_proxy_type::const_iterator                        const_iterator;
        typedef typename type_proxy_type::const_reverse_iterator                const_reverse_iterator;

        iterator begin()
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            //return zen.data_.begin();
            return zen.data_;
        }

        iterator end()
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            return zen.data_ + zen.size();
            //return zen.data_.end();
        }

        const_iterator begin() const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            //return zen.data_.begin();
            return zen.data_;
        }

        const_iterator end() const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            //return zen.data_.end();
            return zen.data_ + zen.size();
        }

        const_iterator cbegin() const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            //return zen.data_.begin();
            return zen.data_;
        }

        const_iterator cend() const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            //return zen.data_.end();
            return zen.data_ + zen.size();
        }

        reverse_iterator rbegin()
        {
            return reverse_iterator( end() );
        }

        reverse_iterator rend()
        {
            return reverse_iterator( begin() );
        }

        const_reverse_iterator rbegin() const
        {
            return const_reverse_iterator( end() );
        }

        const_reverse_iterator rend() const
        {
            return const_reverse_iterator( begin() );
        }

        const_reverse_iterator crbegin() const
        {
            return const_reverse_iterator( end() );
        }

        const_reverse_iterator crend() const
        {
            return const_reverse_iterator( begin() );
        }

    };//struct crtp_direct_iterator

}

#endif//_DIRECT_ITERATOR_HPP_INCLUDED_DSPON43PIOHAFSKLJH4098HASFDLKJNHSDKFJNADFD

