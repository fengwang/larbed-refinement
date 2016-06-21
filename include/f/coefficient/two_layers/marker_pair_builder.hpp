#ifndef GPBBVQXCPWHCQNCSKMJJKFEDDCISXRYHVDPHKRDQYXOSAYYTNKCGJHUYTXKNGFXXXJNWMCSQY
#define GPBBVQXCPWHCQNCSKMJJKFEDDCISXRYHVDPHKRDQYXOSAYYTNKCGJHUYTXKNGFXXXJNWMCSQY

#include <f/coefficient/two_layers/marker_pair.hpp>
#include <f/matrix/matrix.hpp>

#include <set>
#include <map>
#include <cassert>

namespace f
{

    struct marker_pair_builder
    {
        typedef long                                        value_type;
        typedef marker_pair                                 key_type;
        typedef std::set<key_type>                          marker_set_type;
        typedef std::map<key_type, unsigned long>           marker_offset_associate_type;

        marker_set_type                                     collection;
        marker_offset_associate_type                        offset;
        
        //input:
        //          Ar              ug marker matrix
        //          col             the column selected in I, 0 or middle
        //
        //function:
        //          calculate the combinations in two layers C1 simulation
        template< typename T >
        marker_pair_builder( matrix<T> const& Ar, unsigned long col ) 
        {
            assert( col < Ar.col() );

            //construct set
            for ( unsigned long r = 0; r != Ar.row(); ++r )
                for ( unsigned long c = 0; c != Ar.col(); ++c )
                {
                    if ( r == c )
                    {
                        if ( c != col )
                            collection.insert( key_type{ -1, Ar[c][col] } );
                        continue;
                    }
                    if ( c == col )
                    {
                        collection.insert( key_type{ Ar[r][c], -1 } );
                        continue;
                    }
                    collection.insert( key_type{ Ar[r][c], Ar[c][col] } );
                }

            //construct offset
            unsigned long index = 0;
            for ( auto const& key : collection )
                offset[key] = index++;
        }

    };

}//namespace f

#endif//GPBBVQXCPWHCQNCSKMJJKFEDDCISXRYHVDPHKRDQYXOSAYYTNKCGJHUYTXKNGFXXXJNWMCSQY
