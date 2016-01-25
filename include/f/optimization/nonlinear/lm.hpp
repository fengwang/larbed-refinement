#ifndef NBWYVFCKVRNIPYVWILXJSTXJMJDROIHCHGWTQVVJRVJDDLMCADYMISEKSPBBNGNMYRYLUKJFK
#define NBWYVFCKVRNIPYVWILXJSTXJMJDROIHCHGWTQVVJRVJDDLMCADYMISEKSPBBNGNMYRYLUKJFK

#include <f/matrix/matrix.hpp>
#include <f/derivative/derivative.hpp>
#include <f/derivative/second_derivative.hpp>
#include <f/matrix/numeric/lu_solver.hpp>
#include <f/matrix/numeric/conjugate_gradient_squared.hpp>

#include <cassert>

#include <iostream>

namespace f
{
    struct lm
    {

        template<typename Function>
        auto operator()( Function chi, matrix<double>& a, unsigned long n, double lambda, matrix<double>& alpha, matrix<double> const& beta, double eps, unsigned long max_loop, double chi_chi ) const
        {
            if ( std::isnan( lambda ) || std::isinf( lambda ) ) 
            {
                std::cout << "\nlm: nan/inf found in factor lambda, exit\n";
                return chi_chi;
            }

            if ( !max_loop )
            {
                std::cout << "\nmax loop reached.\n";
                return chi_chi;
            }


            matrix<double> alpha_{ alpha };
            std::for_each( alpha_.diag_begin(), alpha_.diag_end(), [lambda]( double& x ){ x *= lambda + 1.0; } );

            matrix<double> a_{ n, 1 };

            //if failed to solve the function, return residual and exit lm
            if ( lu_solver( alpha_, a_, beta ) )
                if ( conjugate_gradient_squared( alpha_, a_, beta ) )
                {
                    std::cout << "\nfailed to solve the equation, exit\n";
                    std::cout << "alpha is\n" << alpha_ << "\n";
                    std::cout << "beta is\n" << beta << "\n";
                    std::cout << std::endl;
                    return chi_chi;
                }

            a_ += a;
            double const chi_chi_ = chi( a_.data() );

            if ( chi_chi_ >= chi_chi )
                return lm{}( chi, a, n, lambda * 1.64872127070012814685, alpha,  beta, eps, max_loop-1, chi_chi );

            a.swap( a_ );

            //std::cout << chi_chi_ << "\t\t";
            //std::cout << a.transpose() << "\n";

            if ( chi_chi / chi_chi_ - 1.0 < eps ) 
            {
                std::cout << "\neps reached, exit\n";
                std::cout << std::endl;
                return chi_chi_;
            }

            return lm{}( chi, a, n, lambda / 1.57079632679489661923, eps, max_loop-1, chi_chi_ );
        }

        template<typename Function>
        auto operator()( Function chi, matrix<double>& a, unsigned long n, double lambda, double eps, unsigned long max_loop, double chi_chi ) const
        {
            assert( n );

            if ( ! a.row() ) a.resize( n, 1 );

            if ( std::isnan(chi_chi) ) chi_chi = chi( a.data() );

            if ( !max_loop )
            {
                std::cout << "\nmax loop reached.\n";
                std::cout << std::endl;
                return chi_chi;
            }

            if ( std::isnan( lambda ) || std::isinf( lambda ) ) 
            {
                std::cout << "\nnan/inf found, exit\n";
                std::cout << std::endl;
                return chi_chi;
            }

            //generate alpha
            matrix<double> alpha( n, n );
            for ( unsigned long r = 0; r != n; ++r )
                for ( unsigned long c = 0; c  <= r; ++c )
                {
                    auto const& df_rc = make_second_derivative( chi, r, c );
                    alpha[r][c] = df_rc( a.data() );
                    alpha[c][r] = alpha[r][c];
                }

            //generate beta
            matrix<double> beta( n, 1 );
            for ( unsigned long r = 0; r != n; ++r )
            {
                auto const& df_r = make_derivative( chi, r );
                beta[r][0] = - df_r( a.data() );
            }

            return lm{}( chi, a, n, lambda, alpha, beta, eps, max_loop, chi_chi );
        }

    };// struct

    template<typename Function>
    auto make_lm( Function chi, matrix<double>& a, unsigned long n, double lambda = 0.001, double eps = 1.0e-15, unsigned long max_loop = 1000 )
    {
        return lm{}( chi, a, n, lambda, eps, max_loop, std::sqrt(-1) );
    }

}//namespace f

#endif//NBWYVFCKVRNIPYVWILXJSTXJMJDROIHCHGWTQVVJRVJDDLMCADYMISEKSPBBNGNMYRYLUKJFK

