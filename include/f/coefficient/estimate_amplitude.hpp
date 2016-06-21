#ifndef RHWINTDDOEIIMHRQSMDMPSQAMQSEKEUKJJLYUQDWBBLVCOPPDHGIWQUGXMVOYLYRCHVPFDADJ
#define RHWINTDDOEIIMHRQSMDMPSQAMQSEKEUKJJLYUQDWBBLVCOPPDHGIWQUGXMVOYLYRCHVPFDADJ

#include <f/coefficient/coefficient.hpp>
#include <f/coefficient/expm.hpp>
#include <f/matrix/matrix.hpp>

#include <cassert>
#include <complex>
#include <cstddef>
#include <cmath>
#include <map>
#include <vector>
#include <iostream>
#include <iterator>
#include <algorithm>

namespace f
{

    template< typename T>
    struct estimate_amplitude
    {
        typedef T                                   value_type;
        typedef std::complex<value_type>            complex_type;
        typedef std::vector<value_type>             array_type;
        typedef std::vector<complex_type>           complex_array_type;
        typedef std::size_t                         size_type;
        typedef std::map<size_type, array_type>     order_array_type;
        typedef matrix<size_type>                   size_matrix_type;

        order_array_type            record;

        void append( complex_array_type const& diag, size_matrix_type const& marker, complex_type const& t, size_type column, array_type const& I )
        {
            append( diag.begin(), diag.end(), marker, t, column, I );
        }
    
        template< typename Itor >
        void append( Itor diag_begin, Itor diag_end, size_matrix_type const& marker, complex_type const& t, size_type column, array_type const& I )
        {
            append( diag_begin, diag_end, marker, t, column, I.begin() );
        }
    
        template< typename Itor, typename I_Itor >
        void append( Itor diag_begin, Itor diag_end, size_matrix_type const& marker, complex_type const& t, size_type column, I_Itor i_begin )
        {
            assert( marker.row() == marker.col() );
            assert( std::distance( diag_begin, diag_end ) == marker.col() );
            assert( column < marker.col() );
            
            size_type const n = marker.row();

            coefficient<value_type> cof( t, diag_begin, diag_end );

            for ( size_type i = 0; i != n; ++i )
            {
                auto const& I_i = *i_begin++;

                if ( i == column ) continue;

                value_type amp = std::sqrt(I_i/std::norm(cof(i,column)));
                auto& arr = record[marker[i][column]];
                arr.push_back( amp );
            }
        }

        void dump() const
        {
            /*
            size_type const total = record.size();

            for ( size_type i = 1; i != total+1; ++i )
            {
                auto itor = record.find(i);
                if ( itor == record.end() )
                {
                    std::cout << "\nUg_" << i << " is null.\n";
                    continue;
                }

                auto const& arr = (*itor).second;
                if ( 0 == arr.size() )
                {
                    std::cout << "\nFailed to calculate U_" << i << " amplitude.\n";
                    continue;
                }

                auto const average = std::accumulate( arr.begin(), arr.end(), value_type(0) ) / arr.size();

                std::cout << "\nU_" << i << " is " << average << "\n";
            }
            */

            for ( auto itor = record.begin(); itor != record.end(); ++itor )
            {
                auto i = (*itor).first;
                auto const& arr = (*itor).second;

                auto const average = std::accumulate( arr.begin(), arr.end(), value_type(0) ) / arr.size();

                std::cout << "\nU_" << i << " is " << average <<  " ---- size is " << arr.size() << "\n";
            }
        }

    
    };//struct estimate_amplitude

}//namespace f

#endif//RHWINTDDOEIIMHRQSMDMPSQAMQSEKEUKJJLYUQDWBBLVCOPPDHGIWQUGXMVOYLYRCHVPFDADJ

