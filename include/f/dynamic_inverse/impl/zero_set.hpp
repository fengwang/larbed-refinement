#ifndef TNKFCLBHBJTELQRFUVAXWCRBRUVCGOVXNDYFVQALHIVDTEAFQQKHEJAUAHFVPJGOCKIHVNEOE
#define TNKFCLBHBJTELQRFUVAXWCRBRUVCGOVXNDYFVQALHIVDTEAFQQKHEJAUAHFVPJGOCKIHVNEOE

#include <set>
#include <iostream>
#include <iterator>
#include <cmath>
#include <algorithm>

namespace f
{
    template< typename T, typename Zen >
    struct zero_set
    {
        typedef T                                       value_type;
        typedef Zen                                     zen_type;
        typedef unsigned long                           size_type;
        typedef std::set<size_type>                     zero_set_type;

        value_type                                      eps_;
        zero_set_type                                   zero_set_;

        void config_zero_set_eps( value_type const eps )
        {
            eps_ = eps;
        }

        value_type zero_set_eps() const
        {
            return eps_;
        }

        void make_zero_set()
        {
            make_zero_set( eps_ );
        }

        void make_zero_set( value_type const eps )
        {
            zen_type const& zen = static_cast<zen_type const&>(*this);

            if ( zen.first_order_approximation_result_size() == 0 )
                std::cerr << "\nmust make_first_order_approximation before calling make_zero_set.\n";

            for ( size_type index = 1; index != zen.ug_size(); ++index )
                if ( std::abs( zen.first_order_approximation_result(index) ) < eps )
                {
                    zero_set_.insert( index );
                }
        }

        bool is_zero( size_type index ) const
        {
            if ( zero_set_.find(index) != zero_set_.end() )
                return true;
            return false;
        }

        void dump_zero_set() const
        {
            std::cout << "\nzero_set::dump_zero_set:\n";
            std::cout << std::hex;
            std::copy( zero_set_.begin(), zero_set_.end(), std::ostream_iterator<size_type>( std::cout, "\t" ) );
            std::cout << "\n";
        }
    };

}//namespace f

#endif//TNKFCLBHBJTELQRFUVAXWCRBRUVCGOVXNDYFVQALHIVDTEAFQQKHEJAUAHFVPJGOCKIHVNEOE

