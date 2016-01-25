#ifndef SANDNDYQQITVPTVKBOHHNANPKQUVDEVGOHWRXIVCQGLFPJXSIDCTNCLMLFFVEPTOQNKJEMLIG
#define SANDNDYQQITVPTVKBOHHNANPKQUVDEVGOHWRXIVCQGLFPJXSIDCTNCLMLFFVEPTOQNKJEMLIG

#include <f/matrix/matrix.hpp>
#include <f/matrix/numeric/lu_solver.hpp>
#include <f/matrix/numeric/conjugate_gradient_squared.hpp>
#include <f/derivative/derivative.hpp>

#include <algorithm>
#include <functional>
#include <cstddef>

//WRONG

namespace f
{

    template< typename T >
    struct line_search
    {
        typedef T                                               value_type;
        typedef value_type*                                     pointer;
        typedef std::function<value_type(pointer)>              function_type;
        typedef std::size_t                                     size_type;

        function_type                                           merit_function;
        matrix<function_type>                                   derivative_function; //\delta f -- [n][1]
        size_type                                               variable_length;
        size_type                                               total_step;
        size_type                                               current_step;
        matrix<value_type>                                      direction;
        matrix<value_type>                                      gradient_k;
        matrix<value_type>                                      gradient_k_1;
        matrix<value_type>                                      initial_guess;      // [n][1]
        matrix<value_type>                                      current_solution;   // [n][1]
        matrix<value_type>                                      trial_solution;   // [n][1]
        value_type                                              step_size;
        value_type                                              eps;
        value_type                                              current_residual;

        std::function<void(value_type, pointer)>                iteration_over_function;

        void make_step_size() //default
        {
            auto const& interpolate = []( value_type x1, value_type y1, value_type x2, value_type y2, value_type x3, value_type y3 )
            {
                matrix<value_type> A( 3, 3 );
                matrix<value_type> x( 3, 1 );
                matrix<value_type> B( 3, 1 );
                A[0][0] = x1 * x1; A[0][1] = x1; A[0][2] = value_type{1}; B[0][0] = y1;
                A[1][0] = x2 * x2; A[1][1] = x2; A[1][2] = value_type{1}; B[1][0] = y2;
                A[2][0] = x3 * x3; A[2][1] = x3; A[2][2] = value_type{1}; B[2][0] = y3;

                int const status = lu_solver( A, x, B );
                if ( 1 == status ) conjugate_gradient_squared( A, x, B );

                return x[1][0] / ( x[0][0]+x[0][0] ); 
            };

            auto const& phi = [this]( value_type alpha ) 
            { 
                (*this).current_solution += alpha * (*this).direction;
                auto const ans = (*this).merit_function( (*this).current_solution.data() );
                (*this).current_solution -= alpha * (*this).direction;
                return ans;
            };
            //auto const& d_phi = make_derivative( phi, 0 );
            auto const& d_phi = make_derivative<0>( phi );

            value_type const c_1 = 0.00186744273170798882; // e^{-2pi}
            //value_type const c_2 = value_type{1} - c_1;
            value_type const c_2 = 0.9;
            value_type const alpha_max = 1.4142135623730950488; // sqrt(2)
            value_type const ratio = 0.6180339887498948482;
            value_type const alpha_next = alpha_max * ratio;

            value_type const phi_0 = phi( value_type{0} );
            value_type const d_phi_0 = d_phi( value_type{0} );
            value_type const phi_alpha_max = phi( alpha_max );

            value_type alpha = interpolate( value_type{0}, phi_0, alpha_next, phi(alpha_next), alpha_max, phi_alpha_max );

            if ( std::isnan( alpha ) || std::isinf( alpha ) ) alpha = alpha_next;
            if ( alpha > alpha_max || alpha <= value_type{0} ) alpha = alpha_next;

            value_type phi_last_alpha = phi_0;
            value_type last_alpha = 0;
            value_type phi_alpha = phi( alpha );

            auto const& zoom = [interpolate, ratio, c_1, c_2, phi_0, d_phi_0, phi, d_phi]( value_type alpha_low, value_type phi_alpha_low, value_type alpha_high, value_type phi_alpha_high  )
            {
                auto recurser = [interpolate, ratio, c_1, c_2, phi_0, d_phi_0, phi, d_phi]( value_type alpha_low, value_type phi_alpha_low, value_type alpha_high, value_type phi_alpha_high, auto recur )
                {
                    value_type const alpha_se = alpha_low + ( alpha_high - alpha_low ) * ratio;
                    if ( std::abs( alpha_low - alpha_high ) < value_type( 1.0e-10 ) ) return alpha_se;

                    value_type const phi_alpha_se = phi( alpha_se );
                    value_type alpha_j = interpolate( alpha_low, phi_alpha_low, alpha_se, phi_alpha_se, alpha_high, phi_alpha_high );
                    if ( alpha_j > std::max( alpha_low, alpha_high ) || alpha_j < std::min( alpha_low, alpha_high ) ) alpha_j = alpha_se;
                    if ( std::isinf( alpha_j ) || std::isnan( alpha_j ) ) alpha_j = alpha_se;
                    value_type const phi_alpha_j = phi( alpha_j );

                    value_type const d_phi_alpha_j = d_phi( alpha_j );
                    if ( std::abs(d_phi_alpha_j) <= -c_2 * d_phi_0 ) 
                        return alpha_j;

                    if ( ( phi_alpha_j > phi_0 + c_1 * alpha_j * d_phi_0 ) | ( phi_alpha_j >= phi_alpha_low ) )
                        return recur( alpha_low, phi_alpha_low, alpha_j, phi_alpha_j, recur );

                    if ( d_phi_alpha_j * ( alpha_high - alpha_low ) >= value_type{0} )
                        return recur( alpha_j, phi_alpha_j, alpha_low, phi_alpha_low, recur );

                    return recur( alpha_j, phi_alpha_j, alpha_high, phi_alpha_high, recur );
                };

                return recurser( alpha_low, phi_alpha_low, alpha_high, phi_alpha_high, recurser );
            };

            for ( size_type index = 0; true; ++index )
            {
                if ( index != 0 ) 
                    phi_alpha = phi( alpha );

                if ( ( ( phi_alpha > phi_0 + c_1 * alpha * d_phi_0 ) && ( phi_alpha < phi_0 ) ) || ( index != 0  && phi_alpha >= phi_last_alpha ) )
                {
                    step_size = zoom( last_alpha, phi_last_alpha, alpha, phi_alpha );
                    break;
                }

                value_type const d_phi_alpha = d_phi( alpha );
                if ( ( std::abs(d_phi_alpha) <= -c_2 * d_phi_0 ) && ( phi_alpha < phi_0 ) )
                {
                    step_size = alpha;
                    break;
                }

                if ( d_phi_alpha >= 0 )
                {
                    step_size = zoom( alpha, phi_alpha, last_alpha, phi_last_alpha );
                    break;
                }

                value_type const alpha_se = alpha + ( alpha_max - alpha ) * ratio;
                value_type const phi_alpha_se = phi( alpha_se );
                alpha = interpolate( alpha, phi_alpha, alpha_se, phi_alpha_se, alpha_max, phi_alpha_max );
                phi_last_alpha = phi_alpha;
            }

            //current_solution += step_size * direction;
            current_solution -= step_size * direction;
            current_residual = merit_function( current_solution.data() );

            std::cerr << "\n\nstep is set to " << step_size;
            std::cerr << "\nresidual is  " << current_residual << "\n";
            std::cerr << "direction is " << direction.transpose() << "\n";
            std::cerr << "solution is " << current_solution.transpose() << "\n";
        }

        template< typename Function >
        line_search( Function const& func_, size_type len_, size_type loop_ = 1000 ) : merit_function{ func_ }, variable_length{ len_ }
        {
            derivative_function.resize( variable_length, 1 );

            for ( size_type index = 0; index != variable_length; ++index )
                derivative_function[index][0] = make_derivative( merit_function, index );

            current_step = 0;
            total_step = loop_;

            initial_guess.resize( variable_length, 1 );
            std::fill( initial_guess.begin(), initial_guess.end(), value_type{} );

            current_solution = initial_guess;

            direction.resize( variable_length, 1 );

            eps = 1.0e-10;

            gradient_k.resize( variable_length, 1 );
            gradient_k_1.resize( variable_length, 1 );

            iteration_over_function = [](value_type,pointer){};
        }

        template< typename F >
        void config_iteration_function( F func )
        {
            iteration_over_function = func;
        }

        void iterate()
        {
            make_direction();
            make_step_size();

            //current_solution += step_size * direction;
            ++current_step;

            iteration_over_function( current_residual, current_solution.data() );
        }

        template< typename Otor >
        value_type operator()( Otor oi_ )
        {
            current_solution = initial_guess;
            current_residual = merit_function( current_solution.data() );

            while ( current_step != total_step )
            {
                iterate();
                value_type const direction_norm = std::inner_product( direction.begin(), direction.end(), direction.begin(), value_type{} );

                if ( std::isnan(direction_norm) || std::isinf(direction_norm) ) break;

                if ( direction_norm < eps ) break;
            }
            std::copy( current_solution.begin(), current_solution.end(), oi_ );
            return merit_function( &(current_solution[0][0]) );
        }

        void make_direction() //default
        {
            /*
            for ( size_type i = 0; i != variable_length; ++i )
                direction[i][0] = -derivative_function[i][0]( current_solution.data() );
            */

            gradient_k.swap( gradient_k_1 );

            for ( size_type i = 0; i != variable_length; ++i )
                gradient_k_1[i][0] = derivative_function[i][0](current_solution.data() );

            if ( 0 == current_step )
            {
                direction = -gradient_k_1;
                return;
            }

            value_type const beta_k1_k1 = std::inner_product( gradient_k_1.begin(), gradient_k_1.end(), gradient_k_1.begin(), value_type{} );
            value_type const beta_k_p = std::inner_product( gradient_k.begin(), gradient_k.end(), direction.begin(), value_type{} );
            value_type const beta_k1_p = std::inner_product( gradient_k_1.begin(), gradient_k_1.end(), direction.begin(), value_type{} );
            value_type const beta = beta_k1_k1 / ( beta_k1_p - beta_k_p );

            direction *= beta;
            direction -= gradient_k_1;
        }

        template< typename Function >
        void config_derivative_function( Function const& df_, size_type index_ )
        {
            assert( index_ < variable_length );
            derivative_function[index_][0] = df_;
        }

        template< typename Itor >
        void config_initial_guess( Itor begin_ )
        {
            for ( size_type index = 0; index != variable_length; ++index )
            {
                initial_guess[index][0] = *begin_;
                ++begin_;
            }

            current_step = 0;
            current_solution = initial_guess;
        }

        void config_initial_guess( value_type x )
        {
            initial_guess[0][0] = x;
            current_step = 0;
            current_solution = initial_guess;
        }

        void config_total_steps( size_type loop_ )
        {
            total_step = loop_;
        }

        void config_eps( value_type const eps_ )
        {
            eps = eps_;
        }

    };//struct line_search

}//namespace f

#endif//SANDNDYQQITVPTVKBOHHNANPKQUVDEVGOHWRXIVCQGLFPJXSIDCTNCLMLFFVEPTOQNKJEMLIG

