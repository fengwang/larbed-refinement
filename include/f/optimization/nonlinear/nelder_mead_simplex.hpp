#ifndef ONJUHFHMOJCPDDDVUBEBJRGHKJPKXVVJRHNWRHMDWDNGVSQDCMBQECBEVBHKMAYLLXCWGKERY
#define ONJUHFHMOJCPDDDVUBEBJRGHKJPKXVVJRHNWRHMDWDNGVSQDCMBQECBEVBHKMAYLLXCWGKERY

#include <f/matrix/matrix.hpp>
#include <f/derivative/derivative.hpp>

#include <algorithm>
#include <functional>
#include <cstddef>
#include 

namespace f
{

    template< typename T, typename Concret_Descent >
    struct nelder_mead_simplex
    {
        typedef T                                               value_type;
        typedef Concret_Descent                                 zen_type;
        typedef value_type*                                     pointer;
        typedef std::function<value_type(pointer)>              function_type;
        typedef std::size_t                                     size_type;

        nelder_mead_simplex() noexcept : alpha( 1.0 ), beta( 0.5 ), gamma( 2.0 ), delta( 0.5 ) {}

        function_type                                           merit_function;
        size_type                                               unknown_parameters;

        value_type                                              alpha;
        value_type                                              beta;
        value_type                                              gamma;
        value_type                                              delta;

        //
        // data_cache[unknown_parameters+4,unknown_parameters+1]
        // the evaluated parameter are stored at the last postion of each line
        //
        // [ *, ....., *]  <-- 0                        -->>    P_low
        // [ *, ....., *]  <-- 1                        -->>    P_high_se
        // [ .........  ]  <-- 2  
        //  
        // [ .........  ]
        // [ *, ....., *]  <--unknown_parameters        -->>    P_high
        // [ *, ....., *]  <--unknown_parameters+1      -->>    P*
        // [ *, ....., *]  <--unknown_parameters+2      -->>    P**
        // [ *, ....., *]  <--unknown_parameters+3      -->>    P^-
        //
        matrix<value_type>                                      data_cache;

        int iterate() noexcept
        {
            make_order();
            make_centroid();

            make_reflection();

            if ( data_cache[unknown_parameters+3][unknown_parameters] < data_cache[0][unknown_parameters] )
            {
                make_expansion();

                if ( data_cache[unknown_parameters+2][unknown_parameters] < data_cache[0][unknown_parameters] )
                {
                    make_replacement_oo();
                    return 1;
                }

                make_replacement_o();
                return 2;
            }

            if ( data_cache[unknown_parameters+1][unknown_parameters] > data_cache[1][unknown_parameters] )
            {
                if ( data_cache[unknown_parameters+1][unknown_parameters] <= data_cache[unknown_parameters][unknown_parameters] )
                    make_replacement_o();

                make_contraction();

                if ( data_cache[unknown_parameters+2][unknown_parameters] > data_cache[unknown_parameters][unknown_parameters] )
                {
                    make_reduction();
                    return 5;
                }

                make_replacement_oo();
                return 4;
            }

            make_replacement_o();
            return 3;
        }

        void make_evaluation( unsigned long const index ) noexcept
        {
            assert( index < unknown_parameters+3 );
            data_cache[index][unknown_parameters] = merit_function( &data_cache[index][0] );
        }

        void make_replacement_o() noexcept
        {
            std::swap_ranges( data_cache.row_begin(unknown_parameters), data_cache.row_end(unknown_parameters), data_cache.row_begin(unknown_parameters+1) );
        }

        void make_replacement_oo() noexcept
        {
            std::swap_ranges( data_cache.row_begin(unknown_parameters), data_cache.row_end(unknown_parameters), data_cache.row_begin(unknown_parameters+2) );
        }

        void operator()() noexcept
        {
            make_evaluation();

            for (;;)
            {
                iterate();
                if ( is_final() ) 
                    break;
            }
        }

        //
        // evaluate all elements in data_cache,
        void make_evaluation() noexcept
        {
            for ( size_type r = 0; r != unknown_parameters+1; ++r )
                make_evaluation( r );
        }

        //
        // rearrayge element in data_cache, place P_Low, P_low_se and P_high in place
        void make_order() noexcept
        {
            //find min
            size_type const offset_low = std::min_element( data_cache.col_begin(unknown_parameters), data_cache.col_begin(unknown_parameters)+unknown_parameters+1 ) - data_cache.col_begin(unknown_parameters);
            std::swap_ranges( data_cache.row_begin(0), data_cache.row_end(0), data_cache.row_begin(offset_low) );
            //find max
            size_type const offset_max = std::max_element( data_cache.col_begin(unknown_parameters), data_cache.col_begin(unknown_parameters)+unknown_parameters+1 ) - data_cache.col_begin(unknown_parameters);
            std::swap_ranges( data_cache.row_begin(unknown_parameters), data_cache.row_end(unknown_parameters), data_cache.row_begin(offset_max) );
            //find max_se
            size_type const offset_max_se = std::max_element( data_cache.col_begin(unknown_parameters), data_cache.col_begin(unknown_parameters)+unknown_parameters ) - data_cache.col_begin(unknown_parameters);
            std::swap_ranges( data_cache.row_begin(1), data_cache.row_end(1), data_cache.row_begin(offset_max_se) );
        }

        // generate P^-
        void make_centroid() noexcept
        {
            for( size_type c = 0; c != unknown_parameters; ++c )
                data_cache[unknown_parameters+3][c] = std::accumulate( data_cache.col_begin(c), data_cache.col_begin(c)+unknown_parameters, value_type(0) ) / static_cast<value_type>( unknown_parameters );

            make_evaluation( unknown_parameters+3 );
        }

        // P* = P^- + alpha( P^- - Ph )
        void make_reflection() noexcept
        {
            std::transform( data_cache.row_begin(unknown_parameters+3), data_cache.row_end(unknown_parameters+3), data_cache.row_begin(unknown_parameters), data_cache.row_begin(unknown_parameters+1),
                            [this]( value_type a, value_type b ) { value_type const factor = (*this).alpha; return a + factor * ( a - b ); } );
        }

        // P** = P* + gamma( P* - P^- )
        void make_expansion() noexcept
        {
            std::transform( data_cache.row_begin(unknown_parameters+1), data_cache.row_end(unknown_parameters+1), data_cache.row_begin(unknown_parameters+3), data_cache.row_begin(unknown_parameters+2),
                            [this]( value_type a, value_type b ) { value_type const factor = (*this).gamma; return a + factor * ( a - b ); } );
        }

        // P** = P^- + beta( Ph - P^- )
        void make_contraction() noexcept
        {
            std::transform( data_cache.row_begin(unknown_parameters+3), data_cache.row_end(unknown_parameters+3), data_cache.row_begin(unknown_parameters), data_cache.row_begin(unknown_parameters+1),
                            [this]( value_type a, value_type b ) { value_type const factor = (*this).beta; return a + factor * ( a - b ); } );
        }

        // Pi = ( Pi + Pl ) / 2
        void make_reduction() noexcept
        {
            for ( size_type r = 1; r != unknown_parameters+1; ++r )
            {
                std::transform( data_cache.row_begin(0), data_cache.row_end(0), data_cache.row_begin(r), data_cache.row_begin(r)
                                [this]( value_type a, value_type b ){ value_type const factor = (*this).delta; return a + factor * ( b - a ); } );
                make_evaluation( r );
            }

        }

        bool is_final() noexcept
        {
            return false;
        }


    };//struct nelder_mead_simplex 

}//namespace f

#endif//ONJUHFHMOJCPDDDVUBEBJRGHKJPKXVVJRHNWRHMDWDNGVSQDCMBQECBEVBHKMAYLLXCWGKERY

