#ifndef PCMCHXRBBDUCJPJMVQJRASRVIVSSMOAWKFBEABMKPCMEWLVRGUQQMBLRQGBDWQNQMROESWPDJ
#define PCMCHXRBBDUCJPJMVQJRASRVIVSSMOAWKFBEABMKPCMEWLVRGUQQMBLRQGBDWQNQMROESWPDJ

#include <f/matrix/matrix.hpp>
#include <f/derivative/derivative.hpp>

#include <algorithm>
#include <functional>
#include <cstddef>

namespace f
{

    template< typename T, typename Concret_Algorithm >
    struct step_direction_iteration
    {
        typedef T                                               value_type;
        typedef Concret_Algorithm                               zen_type;
        typedef value_type*                                     pointer;
        typedef std::function<value_type(pointer)>              function_type;
        typedef std::function<void(pointer,size_type)>          procedure_function_type;
        typedef std::size_t                                     size_type;
        typedef matrix<value_type>                              matrix_type;

        function_type                                           merit_function;
        size_type                                               unknowns;
        std::function<bool(value_type)>                         residual_threshold_function;
        std::function<bool(size_type)>                          step_threshold_function;

        size_type                                               step_counter;
        value_type                                              current_residual;

        matrix_type                                             current_step;
        matrix_type                                             current_direction;
        matrix_type                                             current_solution;

        procedure_function_type                                 on_select_step_begin_function;
        procedure_function_type                                 on_select_step_end_function;

        procedure_function_type                                 on_select_direction_begin_function;
        procedure_function_type                                 on_select_direction_end_function;

        procedure_function_type                                 on_iteration_begin_function;
        procedure_function_type                                 on_iteration_end_function;




    };

    template< typename T, typename Concret_Descent >
    struct default_steepest_descent
    {
        typedef T                                               value_type;
        typedef Concret_Descent                                 zen_type;
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
            value_type const max_step_size = 1.6180339887498948482;
            value_type const decrease_ratio = 0.6180339887498948482;
            step_size = max_step_size;

            for (;;) 
            {
                trial_solution = current_solution + step_size * direction;
                value_type const new_residual = merit_function( trial_solution.data() );
                if ( new_residual < current_residual || step_size < eps )
                {
                    current_residual = new_residual;
                    break;
                }
                step_size *= decrease_ratio;
            }

        }

        template< typename Function >
        default_steepest_descent( Function const& func_, size_type len_, size_type loop_ = 1000 ) : merit_function{ func_ }, variable_length{ len_ }
        {
            //default derivatives
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
            auto& zen = static_cast<zen_type&>(*this);
            zen.make_direction(); //
            zen.make_step_size(); //

            current_solution += step_size * direction;
            ++current_step;

            iteration_over_function( current_residual, current_solution.data() );
#if 0 
            std::cout << "\n\nstep is " << step_size;
            std::cout << "\ndirection is " << direction.transpose();
            std::cout << "\nresidual is " << current_residual;
            std::cout << "\ncurrent_solution is " << current_solution.transpose();
            std::cerr << "\nresidual is " << current_residual;
            //std::ofstream ofs( "solution.noisy.stro3_select_7_5.dat", std::fstream::app );
            //std::ofstream ofs( "solution.sto_accr.12.2.dat", std::fstream::app );
            //std::ofstream ofs( "solution.sto_accr.12.2.se.dat", std::fstream::app );
            //std::ofstream ofs( "solution.sto_accr.12.2.exp.dat", std::fstream::app );
            //std::ofstream ofs( "solution.9beam.12.2.dat", std::fstream::app );
            //std::ofstream ofs( "./solution.sto_accr.12.2.dat", std::fstream::app );
            //std::ofstream ofs( "./solution.sto_accr.12.2_abs_dec_1.dat", std::fstream::app );
            std::ofstream ofs( "./solution.sto_accr.12.2_thickness_dec_2.dat", std::fstream::app );
            //std::ofstream ofs( "solution.9beam.12.2.dat", std::fstream::app );
            //std::ofstream ofs( "new.solution.9beam.12.2.dat", std::fstream::app );
            //std::ofstream ofs( "solution.71.simple.derivative.10.0.dat", std::fstream::app );
            //std::ofstream ofs( "solution.9beams.derivative.10.0.dat", std::fstream::app );
            //std::ofstream ofs( "solution.9beams.derivative.10.0.dat", std::fstream::app );
            ofs << current_residual << "\t\t" << current_solution.transpose() << "\n";
            ofs.close();
#endif
        }

        template< typename Otor >
        value_type operator()( Otor oi_ )
        {
            auto& zen = static_cast<zen_type&>(*this);

            current_solution = initial_guess;
            current_residual = merit_function( current_solution.data() );

            while ( current_step != total_step )
            {
                zen.iterate();
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

    };//struct default_steepest_descent

    template< typename T >
    struct steepest_descent : default_steepest_descent< T, steepest_descent<T> >
    {
        typedef std::size_t                                     size_type;

        template< typename Function >
        steepest_descent( Function const& func_, size_type len_, size_type loop_ = 1000 ) : default_steepest_descent< T, steepest_descent<T> >( func_, len_, loop_ ) {}
    };//struct steepest_descent 

    template< typename T >
    struct direct_steepest_descent : default_steepest_descent< T, steepest_descent<T> >
    {
        typedef std::size_t                                     size_type;
        typedef T                                               value_type;
        typedef value_type*                                     pointer;
        matrix<value_type>                                      ug_tmp{ (*this).variable_length, 1 };

        template< typename Function >
        direct_steepest_descent( Function const& func_, size_type len_, size_type loop_ = 1000 ) : default_steepest_descent< T, steepest_descent<T> >( func_, len_, loop_ ) 
        {
            ug_tmp.resize( len_, 1 );
#if 0
            for ( size_type index = 0; index != (*this).variable_length; ++index )
                (*this).derivative_function[index][0] = [this, index]( pointer x )
                {
                    size_type n = (*this).variable_length;
                    value_type res = (*this).current_residual;
                    value_type step = value_type{ 1.0e-6 };
                    std::copy( x, x+n, (*this).ug_tmp.begin() );
                    (*this).ug_tmp[index][0] += step;
                    value_type new_res = (*this).merit_function( (*this).ug_tmp.data() );
                    return (new_res - (*this).current_residual) / step;
                };
#endif
        }

    };//struct steepest_descent 

}//namespace f

#endif//PCMCHXRBBDUCJPJMVQJRASRVIVSSMOAWKFBEABMKPCMEWLVRGUQQMBLRQGBDWQNQMROESWPDJ

