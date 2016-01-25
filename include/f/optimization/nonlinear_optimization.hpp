#ifndef DJLFMAKXIASEBAWYUCDIOLNLJNHQXALWPVTAAYYLBLIVGGIVTFFQCLPFTOLMYLXVYHUESQQYI
#define DJLFMAKXIASEBAWYUCDIOLNLJNHQXALWPVTAAYYLBLIVGGIVTFFQCLPFTOLMYLXVYHUESQQYI

#include <f/matrix/matrix.hpp>
#include <f/derivative/derivative.hpp>
#include <f/derivative/second_derivative.hpp>

#include <limits>
#include <functional>
#include <cstddef>
#include <cassert>

namespace f
{
    //make ->> setup ->> config
    template<typename T, typename Zen>
    struct nonlinear_optimization
    {
        typedef Zen                                             zen_type;
        //typedef typename Zen::zen_type                          zen_type;
        typedef std::size_t                                     size_type;
        typedef T                                               value_type;
        typedef value_type*                                     pointer;
        typedef matrix<value_type>                              matrix_type;
        typedef std::function<value_type( pointer )>            function_type;
        typedef matrix<function_type>                           function_matrix_type;

        matrix_type                                             fitting_array;              //store the fitting result
        size_type                                               unknown_parameters;
        size_type                                               max_iteration;
        value_type                                              eps;

        function_type                                           merit_function;             //store the \chi^2
        function_matrix_type                                    jacobian_matrix_function;   //store \partial{\chi^2} / \partial{}
        function_matrix_type                                    hessian_matrix_function;    //store \partial^2{\chi^2} / \partial{}\partial

        int operator()()
        {
            if ( make_preprocessing() )                                                                                 { assert( !"Failed to make preprocessing." );                       return 1; }

            if( make_initialization_preprocessing() )                                                                   { assert( !"Failed to make initialization preprocessing." );        return 1; }

            unknown_parameters = make_unknown_parameters(); if( !unknown_parameters )                                   { assert( !"Failed to make unknown parameters." );                  return 1; }

            fitting_array = make_fitting_array_initial_guess(); if ( fitting_array.size() != unknown_parameters  )      { assert( !"Failed to make initial guess." );                       return 1; }

            max_iteration = make_max_iteration(); if( !max_iteration )                                                  { assert( !"Failed to make max iteration." );                       return 1; }

            eps = make_eps();if( eps < value_type{0} )                                                                  { assert( !"Failed to make eps." );                                 return 1; }

            if ( make_initialization_postprocessing() )                                                                 { assert( !"Failed to make initialization postprocessing." );       return 1; }

            merit_function = make_merit_function(); if( !merit_function )                                               { assert( !"Failed to make merit function." );                      return 1; }

            jacobian_matrix_function.resize( 1, unknown_parameters );
            for ( size_type index = 0; index != unknown_parameters; ++index )
            {
                jacobian_matrix_function[0][index] = make_jacobian_matrix_function( index );
                if ( !jacobian_matrix_function[0][index] )                                                              { assert( "Failed to make jacobian matrix function." );             return 1; }
            }

            hessian_matrix_function.resize( unknown_parameters, unknown_parameters );
            for ( size_type index = 0; index != unknown_parameters; ++index )
                for ( size_type jndex = 0; jndex <= index; ++jndex )
                {
                    hessian_matrix_function[index][jndex] = make_hessian_matrix_function( index, jndex );
                    if ( !hessian_matrix_function[index][jndex] )                                                        { assert( "Failed to make jacobian matrix function." );             return 1; }
                    if ( index != jndex )
                    {
                        hessian_matrix_function[jndex][index] = hessian_matrix_function[index][jndex];
                    }
                }

            if ( make_iteration_preprocessing() )                                                                       { assert( !"Failed to make iteration preprocessing." );             return 1; }

            for ( size_type step_index = 0; step_index != max_iteration; ++step_index )
            {
                if ( make_every_iteration_preprocessing() )                                                             { assert( !"Failed to make every iteration preprocessing." );       return 1; }

                if ( make_iteration() )                                                                                 { assert( !"Failed to make iteration" );                            return 1; }

                if ( ! make_fitting_flag() )                                                                            {                                                        /*eps reached*/break;}

                if ( make_every_iteration_postprocessing() )                                                            { assert( !"Failed to make every iteration postprocessing." );      return 1; }
            }

            if ( make_iteration_postprocessing() )                                                                      { assert( !"Failed to make iteration postprocessing." );            return 1; }

            if ( make_postprocessing() )                                                                                { assert( !"Failed to make postprocessing. " );                    return 1; }

            return 0;
        }

        int make_preprocessing()
        {
            auto& zen = static_cast<zen_type&>( *this );
            return zen.setup_preporcessing();
        }

        int  make_initialization_preprocessing()
        {
            auto& zen = static_cast<zen_type&>( *this );
            return zen.setup_initialization_preprocessing();
        }

        size_type make_unknown_parameters()
        {
            auto& zen = static_cast<zen_type&>( *this );
            return zen.setup_unknown_parameters();
        }

        size_type make_max_iteration()
        {
            auto& zen = static_cast<zen_type&>( *this );
            return zen.setup_max_iteration();
        }

        value_type make_eps()
        {
            auto& zen = static_cast<zen_type&>( *this );
            return zen.setup_eps();
        }

        int make_initialization_postprocessing()
        {
            auto& zen = static_cast<zen_type&>( *this );
            return zen.setup_initialization_postprocessing();
        }

        function_type const make_merit_function()
        {
            auto& zen = static_cast<zen_type&>( *this );
            return zen.setup_merit_function();
        }

        function_type const make_jacobian_matrix_function( size_type index )
        {
            auto& zen = static_cast<zen_type&>( *this );
            return zen.setup_jacobian_matrix_function( index );
        }

        function_type const make_hessian_matrix_function( size_type index, size_type jndex )
        {
            auto& zen = static_cast<zen_type&>( *this );
            return zen.setup_hessian_matrix_function( index, jndex );
        }

        // 0 for success, 1 for failure
        int make_iteration_preprocessing()
        {
            auto& zen = static_cast<zen_type&>( *this );
            return zen.setup_iteration_preprocessing();
        }

        int make_every_iteration_preprocessing()
        {
            auto& zen = static_cast<zen_type&>( *this );
            return zen.setup_every_iteration_preprocessing();
        }

        int make_iteration()
        {
            auto& zen = static_cast<zen_type&>( *this );
            return zen.setup_iteration();
        }

        int make_fitting_flag()
        {
            auto& zen = static_cast<zen_type&>( *this );
            return zen.setup_fitting_flag();
        }

        int make_every_iteration_postprocessing()
        {
            auto& zen = static_cast<zen_type&>( *this );
            return zen.setup_every_iteration_postprocessing();
        }

        int make_iteration_postprocessing()
        {
            auto& zen = static_cast<zen_type&>( *this );
            return zen.setup_iteration_postprocessing();
        }

        int make_postprocessing()
        {
            auto& zen = static_cast<zen_type&>( *this );
            return zen.setup_postprocessing();
        }

        matrix_type const make_fitting_array_initial_guess()
        {
            auto& zen = static_cast<zen_type&>( *this );
            return zen.setup_fitting_array_initial_guess();
        }

        /*
         *
         * default implementation below
         *
         */

        int setup_initialization_postprocessing()
        {
            return 0;
        }

        size_type setup_max_iteration()
        {
            return std::numeric_limits<size_type>::max();
        }

        function_type const setup_merit_function()
        {
            return function_type {};
        }

        function_type const setup_jacobian_matrix_function( size_type index )
        {
            auto& zen = static_cast<zen_type&>( *this );
            assert( index < zen.unknown_parameters );
            auto const& df = make_derivative( zen.merit_function, index );
            return df;
        }

        function_type const setup_hessian_matrix_function( size_type index, size_type jndex )
        {
            auto& zen = static_cast<zen_type&>( *this );
            assert( index < zen.unknown_parameters );
            assert( jndex < zen.unknown_parameters );
            auto const& ddf = make_second_derivative( zen.merit_function, index, jndex );
            return ddf;
        }

        int setup_preporcessing()
        {
            return 0;
        }

        int setup_iteration_preprocessing()
        {
            return 1;
        }

        int setup_every_iteration_preprocessing()
        {
            return 0;
        }

        int setup_iteration()
        {
            return 0;
        }

        int setup_fitting_flag()
        {
            return 1;
        }

        int setup_every_iteration_postprocessing()
        {
            return 0;
        }

        int setup_iteration_postprocessing()
        {
            return 0;
        }

        int setup_postprocessing()
        {
            auto& zen = static_cast<zen_type&>( *this );

            std::cout << "\nthe residual is \t" << zen.merit_function( ( zen.fitting_array ).data() ) << "\n";
            std::cout << "\nthe fitting result is\n" << zen.fitting_array << "\n";
            return 0;
        }

        matrix_type const setup_fitting_array_initial_guess()
        {
            auto& zen = static_cast<zen_type&>( *this );
            matrix_type guess { zen.unknown_parameters, 1 };
            std::fill( guess.begin(), guess.end(), value_type {} );
            return guess;
        }

        size_type setup_unknown_parameters()
        {
            return 0;
        }

        value_type setup_eps()
        {
            return value_type {1.0e-5};
        }

        int setup_initialization_preprocessing()
        {
            return 0;
        }
    };

}//namespace f

//#include <f/optimization/nonlinear/levenberg_marquardt.hpp>

#endif//DJLFMAKXIASEBAWYUCDIOLNLJNHQXALWPVTAAYYLBLIVGGIVTFFQCLPFTOLMYLXVYHUESQQYI

