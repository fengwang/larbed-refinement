#ifndef POKIFKJARYOLKBAETODCLRODVUHALXNUEDMDKBCABFTYFHFDSYYEURFUKGVBYHWRWJRLURXPU
#define POKIFKJARYOLKBAETODCLRODVUHALXNUEDMDKBCABFTYFHFDSYYEURFUKGVBYHWRWJRLURXPU

#include <f/optimization/nonlinear/levenberg_marquardt.hpp>
#include <f/matrix/matrix.hpp>

#include <cstddef>
#include <cassert>
#include <numeric>
#include <iostream>
#include <iomanip>

namespace f
{
    template<typename T>
    T double_square_ug_fitting( matrix<T> const& x, matrix<T> const& y, matrix<double>& ug )
    {
        assert( x.row() == y.row() );
        assert( !(x.col()&1) );
        assert( y.col() == 1 );

        std::size_t const n = x.row();
        std::size_t const m = (x.col() >> 1) - 1;

        auto const& f_xa = [m](T* x, T*a)
        {
            T real_part = x[0];
            T imag_part = x[m+1];

            real_part += std::inner_product( a, a+m, x+1, T{0} );
            imag_part += std::inner_product( a, a+m, x+m+2, T{0} );

            return real_part * real_part + imag_part * imag_part;
        };

        levenberg_marquardt<T> lm;
        lm.config_target_function( f_xa );
        lm.config_unknown_parameter_size( m );
        lm.config_experimental_data_size( n );
        lm.config_x( x );
        lm.config_y( y );
        lm.config_eps( T{1.0e-10} );
        //lm.config_max_iteration( 100 );

        //initial guess?
        if ( ug.row() == m && ug.col() == 1 )
            lm.config_initial_guess( ug );


        //config jacobian matrix here??
        for ( std::size_t index = 0; index != m; ++index )
        {
            auto const& pf = [index, m](T* x, T*a)
            {
                T real_part = T{0};
                T imag_part = T{0};

                if ( std::abs(x[index+1]) > T{1.0e-10} )
                    real_part = x[0] + std::inner_product( a, a+m, x+1, T{0} );

                if ( std::abs(x[index+m+2]) > T{1.0e-10} )
                    imag_part = x[m+1] + std::inner_product( a, a+m, x+m+2, T{0} );

                return T{2} * ( real_part * x[index+1]  + imag_part * x[index+m+2] );
            };

            lm.config_jacobian_matrix( index, pf );
        }

        ug.resize( m, 1 );
        lm( ug.col_begin(0) );


        std::cout.precision(15);
        std::cout << lm.chi_square << std::endl;

        return lm.chi_square;
    }

    template<typename T>
    matrix<T> const double_square_ug_fitting( matrix<T> const& x, matrix<T> const& y )
    {
        matrix<T> ug;
        double_square_ug_fitting( x, y, ug );
        return ug;
    }

}//namespace f

#endif//POKIFKJARYOLKBAETODCLRODVUHALXNUEDMDKBCABFTYFHFDSYYEURFUKGVBYHWRWJRLURXPU

