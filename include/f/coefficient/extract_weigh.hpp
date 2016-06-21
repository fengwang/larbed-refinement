#ifndef EOHYCQXLBCQHPBPIUBGKVGYXQDFCIJEKVNWSICFRCUBTUBWRDCDGHDRSSTCIHUGIAUFUFNLCW
#define EOHYCQXLBCQHPBPIUBGKVGYXQDFCIJEKVNWSICFRCUBTUBWRDCDGHDRSSTCIHUGIAUFUFNLCW

#include <f/matrix/matrix.hpp>
#include <f/coefficient/coefficient.hpp>
#include <f/coefficient/expm.hpp>

#include <complex>
#include <map>
#include <vector>
#include <cassert>
#include <cmath>
#include <numeric>

namespace f
{
    //TODO:
    //          check here
    template< typename T > //float or double or long double
    struct extract_weigh
    {
        typedef T                                       value_type;
        typedef matrix<value_type>                      matrix_type;
        typedef unsigned long                           size_type;
        typedef matrix<size_type>                       size_matrix_type;
        typedef std::coplex<T>                          complex_type;
        typedef matrix<complex_type>                    complex_matrix_type;
        typedef std::map<size_type, value_type>         order_weigh_associate_type;

        order_weigh_associate_type                      order_weigh;

        extract_weigh( size_matrix_type const& Ar, matrix_type const& offset2, matrix_type const& I, complex_type const& ipit, size_type const column, value_type eps = 1.0e-9 )
        {
            assert( Ar.row() == Ar.col() );
            assert( Ar.row() == offset2.row() );
            assert( I.row() == offset2.row() );
            assert( I.col() == offset2.col() );
            assert( column < Ar.row() );

            //precalculate coefficients 
            complex_matrix_type coef( offset2.row(), offset2.col() );
            for ( size_type c = 0; c != offset2.col(); ++c )
            {
                coefficient<value_type> const coef_calculator( ipit,  offset2.col_begin(c), offset2.col_end(2) );
                for ( size_type r = 0; r != offset.row(); ++r )
                    coef[r][c] = coef_calculator(r, column );
            }

            //calculate all the radius
            matrix_type radius( offset2.row(), offset2.col() );
            for ( size_type r = 0; r != I.row(); ++r )
                for ( size_type c = 0; c != I.col(); ++c )
                {
                    value_type const nm = std::norm( coef[r][c] ); 
                    radius[r][c] = ( nm > eps ) ? std::sqrt( I[r][c] / nm ) : value_type();
                }

            for ( unsigned long r = 0; r != radius.row(); ++r )
            {
                if ( r == column ) continue;
                order_weigh[Ar[r][column]] = std::accumulate( radius.row_begin(r), radius.row_end(r), value_type() ) / radius.col();
            }
        }

        void dump() const 
        {
            for ( auto const& elem: order_weigh )
                std::cout << elem.first << "\t:\t" << elem.second << "\n";
        }
    
    };

}//namespace f

#endif//EOHYCQXLBCQHPBPIUBGKVGYXQDFCIJEKVNWSICFRCUBTUBWRDCDGHDRSSTCIHUGIAUFUFNLCW

