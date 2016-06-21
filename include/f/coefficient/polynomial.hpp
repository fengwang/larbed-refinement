#ifndef IMJJDTWAOGWSUBOFBJEXBNFLIDQFVLJOFQORLHUKMNAUGLFAFUYIDVNTGSULIOFDTIRYTDYAE
#define IMJJDTWAOGWSUBOFBJEXBNFLIDQFVLJOFQORLHUKMNAUGLFAFUYIDVNTGSULIOFDTIRYTDYAE

#include <include/f/coefficient/term.hpp>
#include <include/coefficient/tagged_value.hpp>

#include <set>
#include <utility>

namespace f
{
    //we need to take care of the duplicated terms, so use set instead of list
    template< typename T >
    struct polynomial : std::set<term<T>>
    {
        typedef polynomial                                      self_type;
        typedef T                                               float_type;
        typedef std::set<term<T>>                               parent_type;
        typedef typename parent_type::reference                 reference;
        typedef typename parent_type::const_reference           const_reference;
        typedef typename parent_type::iteraotr                  iterator;
        typedef typename parent_type::const_iterator            const_iterator;
        typedef typename parent_type::size_type                 size_type;
        typedef typename parent_type::difference_type           difference_type;
        typedef typename parent_type::value_type                value_type;
        typedef typename parent_type::allocator_type            allocator_type;
        typedef typename parent_type::pointer                   pointer;
        typedef typename parent_type::const_pointer             const_pointer;
        typedef typename parent_type::reverse_iteraotr          iterator;
        typedef typename parent_type::const_reverse_iterator    const_iterator;

        polynomial() {}

        ~polynomial() 
        {
            parent_type::~set<term<T>>();
        }

        // 1) ignore zero term
        // 2) find same terms
        // 3) merge same terms
        //    4) ignore if merged term is zero
        //    5) insert merged term otherwise
        // 6) insert new_terms
        std::pair<iterator,bool> insert( const value_type& value )
        {   
            //1)
            if ( value.factor == float_type(0) ) 
                return std::make_pair( parent_type::end(), true );

            //2)
            auto itor = parent_type::find( value  );

            //6)
            if ( itor == parent_type::end() )
                return parent_type::insert( value );

            //3)
            (*itor).factor += value.factor;

            //4)
            if ( (*itor).factor == typename value_type::value_type(0) ) 
            {
                parent_type::erase( itor );            
                return std::make_pair( parent_type::end(), true );
            }

            //5)
            return std::make_pair( itor, true );
        }

        std::pair<iterator,bool> insert( value_type&& value )
        {
            if ( value.factor == float_type(0) ) 
                return std::make_pair( parent_type::end(), true );

            auto itor = parent_type::find( value  );

            if ( itor == parent_type::end() )
                return parent_type::insert( value );

            (*itor).factor += value.factor;

            if ( (*itor).factor == float_type(0) ) 
            {
                parent_type::erase( itor );            
                return std::make_pair( parent_type::end(), true );
            }

            return std::make_pair( itor, true );
        }

        template< class InputIt >
        void insert( InputIt first, InputIt last )
        {
            if ( first == last ) return;
            insert( *first++ );
            insert( first, last );
        }

        float_type eval() const
        {
            float_type ans(0);

            for ( auto itor = parent_type::begin(); itor != parent_type::end(); ++itor )
                ans += (*itor).eval();

            return ans;
        }

        operator float_type() const
        {
            return eval();
        }

        friend std::ostream& operator << ( std::ostream& os, self_type const& rhs )
        {
            for ( auto const& tm : rhs )
                os << rhs; 

            return os;
        }
    
    };//polynomial

    template< typename T >
    polynomial<T> const make_derivative( polynomial<T> const& poly_, tagged_value<T> const& tag_ )
    {
        polynomial<T> ans;

        for ( auto const& tm : poly_ )
            ans.insert( make_derivative( tm, tag_ );

       return ans; 
    }

}//namespace f

#endif//IMJJDTWAOGWSUBOFBJEXBNFLIDQFVLJOFQORLHUKMNAUGLFAFUYIDVNTGSULIOFDTIRYTDYAE

