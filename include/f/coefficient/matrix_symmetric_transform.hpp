#ifndef SGFIMIDKHSMDVMYPHAHHLSRFTKIIRMBINPQUYGNJUPGXVFBSPYPAYQIKRPRQIVWWWVVXQTNNK
#define SGFIMIDKHSMDVMYPHAHHLSRFTKIIRMBINPQUYGNJUPGXVFBSPYPAYQIKRPRQIVWWWVVXQTNNK

#include <f/matrix/matrix.hpp>

#include <algorithm>
#include <cassert>
#include <map>

namespace f
{

    struct matrix_symmetric_transform
    {
        
        template< typename T >
        void operator ()( matrix<T>& Ar ) const 
        {
            typedef T                                   value_type;
            typedef std::map<value_type, value_type>    associate_type;

            assert( Ar.row() == Ar.col() );

            unsigned long n = Ar.row();

            associate_type record;

            for ( unsigned long r = 0; r != n; ++r )
                for ( unsigned long c = 0; c != r; ++c )
                {
                    value_type small_index = std::min( Ar[r][c], Ar[c][r] );
                    value_type large_index = std::max( Ar[r][c], Ar[c][r] );
                    
                    if ( small_index == large_index ) continue;

                    record[large_index] = small_index;
                }

            //remove chain equality, i.e. replace the second element with the smallest one
            for ( auto& k_v : record )
            {
                value_type value = k_v.second;

                while( record.find( value ) != record.end() )
                    value = record[value]; //record[value] is always larger than value

                k_v.second = value;
            }

            for ( unsigned long r = 0; r != n; ++r )
                for ( unsigned long c = 0; c != n; ++c )
                    if ( record.find( Ar[r][c] ) != record.end() )
                        Ar[r][c] = record[Ar[r][c]];
        }
    
    };

}//namespace f

#endif//SGFIMIDKHSMDVMYPHAHHLSRFTKIIRMBINPQUYGNJUPGXVFBSPYPAYQIKRPRQIVWWWVVXQTNNK
