#ifndef XANOBRAGLCCSAEILGTPEAWOBKMBAJMWCFMJUDSBXUAPBVDMCSHDYUEAVUBFGOGFNHCCYHEXDB
#define XANOBRAGLCCSAEILGTPEAWOBKMBAJMWCFMJUDSBXUAPBVDMCSHDYUEAVUBFGOGFNHCCYHEXDB

#include <f/matrix/matrix.hpp>

#include <f/optimization/nonlinear/lm.hpp>

namespace f
{

    template< typename Merit_Function, typename Homotopy_Function >
    auto make_homotopy( Merit_Function chi, Homotopy_Function hom, matrix<double>& a, unsigned long n,  unsigned long steps = 100, double lambda = 0.001, double eps = 1.0e-15, unsigned long max_loop = 1000 )
    {
        double current_residual = 0.0;
        for ( unsigned long index = 0; index != steps; ++index )
        {
            double const alpha = static_cast<double>( index + 1.0 ) / static_cast<double>(steps);
            auto new_func = [=]( double *x )
            {
                return alpha * chi( x ) + ( 1.0 - alpha ) * hom( x );
            };
            
            current_residual = make_lm( new_func, a, n, lambda, eps, max_loop );
        }

        return current_residual;
    }

}//namespace f

#endif//XANOBRAGLCCSAEILGTPEAWOBKMBAJMWCFMJUDSBXUAPBVDMCSHDYUEAVUBFGOGFNHCCYHEXDB

