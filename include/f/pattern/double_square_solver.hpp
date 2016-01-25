#ifndef MDOUBLE_SQUARE_SOLVER_HPP_INCLUDED_DFPOHASDLKJH439OUHAFDSLKJH49UHASFKFEF
#define MDOUBLE_SQUARE_SOLVER_HPP_INCLUDED_DFPOHASDLKJH439OUHAFDSLKJH49UHASFKFEF

#include <f/least_square_fit/nonlinear/levenberg_marquardt_least_square_fit.hpp>
#include <f/matrix/matrix.hpp>

#include <algorithm>
#include <numeric>
#include <functional>

namespace f
{
    namespace double_square_private
    {
        template<typename T>
        struct double_square
        {
            typedef T                                               value_type;
            typedef std::function<value_type( const value_type* )>  function_type;

            unsigned long n;

            double_square( unsigned long n_ ) : n( n_ ) {}

            function_type const operator()( const value_type* x_ ) const
            {
                const value_type* x = x_;
                auto const& ans = [x, this] ( const value_type * a )
                {
                    unsigned long const n = ( *this ).n;
                    value_type const real = std::inner_product( a, a + n, x, value_type{0} );
                    value_type const imag = std::inner_product( a, a + n, x + n, value_type{0} );
                    value_type const power =  real * real + imag * imag;
                    return power;
                };
                return ans;
            }
        };//struct double_square

    }//namespace double_square_private

    //=====================================================================================
    //Comment:
    //  The Model to fit is:
    //     vy_i = (vx_i_0 * a_0 + vx_i_1 * a_1 + ... + vx_i_m-1 * a_m-1)^2 +
    //            (vx_i_m * a_0 + vx_i_m+1 * a_1 + ... + vx_i_m+m-1 * a_m-1)^2
    //  where vector vy and matrix vx are known, and vector a is to be fitted.
    //  The fitting result will be stored to fit_a.
    //
    //Return:
    //  0 ---- Success
    //  1 ---- Failed
    //=====================================================================================
    template<typename Matrix>
    int double_square_solver( Matrix const& vx, Matrix const& vy, Matrix& fit_a )
    {
        typedef typename Matrix::value_type value_type;
        //n is experimental data set size
        //unsigned long const n = vx.row();
        unsigned long const mm = vx.col();
        //m is unknowns to fit
        unsigned long const m = mm >> 1;
        assert( ( m << 1 ) == mm );//even check
        assert( vy.size() == vx.row() );
        fit_a.resize( m, 1 );
        return levenberg_marquardt_least_square_fit( double_square_private::double_square<value_type>{m}, m, vx.begin(), vx.end(), vy.begin(), vy.end(), fit_a.begin() );
    }

}//namespace f

#endif//_DOUBLE_SQUARE_SOLVER_HPP_INCLUDED_DFPOHASDLKJH439OUHAFDSLKJH49UHASFKFEF

