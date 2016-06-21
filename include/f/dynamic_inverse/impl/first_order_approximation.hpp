#ifndef WFEMOLNSDUOAJDDHKLTWCJODWEFRALRBIXDGXVVLHGHHAYUQPEPOAOGUWFJINRWQNHSBCDMTV
#define WFEMOLNSDUOAJDDHKLTWCJODWEFRALRBIXDGXVVLHGHHAYUQPEPOAOGUWFJINRWQNHSBCDMTV

#include <f/coefficient/coefficient.hpp>
#include <f/matrix/matrix.hpp>

#include <map>
#include <functional>
#include <limits>
#include <iostream>
#include <iterator>

namespace f
{
    template< typename T, typename Zen >
    struct first_order_approximation
    {
        typedef T                                       value_type;
        typedef Zen                                     zen_type;
        typedef unsigned long                           size_type;
        typedef std::map<size_type, value_type>         first_order_approximation_result_type;
        typedef std::function<value_type(value_type)>   weigh_function_type;

        first_order_approximation_result_type           first_order_approximation_;
        weigh_function_type                             weigh_function_;

        void dump_first_order_approximation() const
        {
            std::cout << "\nfirst_order_approximation::dump_first_order_approximation:\n";
            for ( auto const& elem : first_order_approximation_ )
                std::cout << "(" << elem.first << ", " << elem.second << ")\t";
            std::cout << "\n";
        }

        //if found, return absolute value
        //otherwise, return max
        value_type first_order_approximation_result( size_type index ) const
        {
            auto const& itor = first_order_approximation_.find(index);
            if ( itor != first_order_approximation_.end() )
                return (*itor).second;

            return std::numeric_limits<value_type>::max();
        }

        size_type first_order_approximation_result_size() const
        {
            return first_order_approximation_.size();
        }

        template<typename Func>
        void config_first_order_approximation_weigh_function( Func f )
        {
            weigh_function_ = f;
        }

        void make_first_order_approximation()
        {
            //default weigh_function
            if ( !weigh_function_ )
                weigh_function_ = [](value_type ){ return 1.0; };

            struct sum_cache
            {
                value_type  approx_weigh_sum;
                value_type  weigh_sum;
            };

            std::map<size_type, sum_cache>  record;

            zen_type& zen = static_cast<zen_type&>(*this);
            size_type const c_index = zen.column_index();

            for ( size_type pattern_index = 0; pattern_index != zen.total_tilt(); ++pattern_index )
            {
                coefficient<value_type> const coef( zen.ipit(), zen.diag_begin(pattern_index), zen.diag_end(pattern_index) );
                for ( size_type intensity_offset = 0; intensity_offset != zen.dimension(pattern_index); ++intensity_offset )
                {
                    if ( intensity_offset == c_index ) continue; //skip central pattern
                    //value_type const I_ij = zen.intensity( pattern_index, intensity_offset );
                    value_type const I_ij = zen.intensity( intensity_offset, pattern_index );
                    value_type const approx = std::sqrt( I_ij / std::norm( coef( intensity_offset, c_index ) ) );
                    value_type const weigh = weigh_function_( I_ij );

                    if ( approx > value_type{0.1} ) continue;

                    size_type const ug_index = zen.ar( pattern_index, intensity_offset, c_index );
                    if ( record.find(ug_index) != record.end() )
                    {
                        auto& sc = record[ug_index];
                        sc.approx_weigh_sum += approx * weigh;
                        sc.weigh_sum += weigh;
                    }
                    else
                        record[ug_index] = sum_cache{approx*weigh, weigh};
                }//intensity_offset loop
            }//pattern_index loop

            for ( auto const& elem : record )
                first_order_approximation_[elem.first] = elem.second.approx_weigh_sum / elem.second.weigh_sum;
        }

    };//struct first_order_approximation

    template< typename T, typename Zen >
    struct first_order_approximation_mock
    {
        typedef T                                       value_type;
        typedef Zen                                     zen_type;
        typedef unsigned long                           size_type;
        typedef std::map<size_type, value_type>         first_order_approximation_result_type;
        typedef f::matrix<value_type>                   matrix_type;

        first_order_approximation_result_type           first_order_approximation_;

        void dump_first_order_approximation() const
        {
            std::cout << "\nfirst_order_approximation::dump_first_order_approximation:\n";
            std::cout << "\nthere are " << first_order_approximation_result_size() << " approximations presented now.\n";
            for ( auto const& elem : first_order_approximation_ )
                std::cout << "(" << std::hex << elem.first << ", " << elem.second << ")\t";
            std::cout << "\n";
        }

        //if found, return absolute value
        //otherwise, return max
        value_type first_order_approximation_result( size_type index ) const
        {
            auto const& itor = first_order_approximation_.find(index);
            if ( itor != first_order_approximation_.end() )
                return (*itor).second;

            return std::numeric_limits<value_type>::max();
        }

        size_type first_order_approximation_result_size() const
        {
            return first_order_approximation_.size();
        }

        void make_first_order_approximation()
        {
            matrix_type ug{"data/SSTO/ug.txt"};
            for ( size_type i = 1; i != ug.row(); ++i )
                first_order_approximation_[i] = std::hypot( ug[i][0], ug[i][1] );
        }

    };//struct first_order_approximation

}//namespace f

#endif//WFEMOLNSDUOAJDDHKLTWCJODWEFRALRBIXDGXVVLHGHHAYUQPEPOAOGUWFJINRWQNHSBCDMTV

