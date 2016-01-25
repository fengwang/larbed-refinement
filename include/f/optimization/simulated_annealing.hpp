#ifndef PAQKLTINTWOCIQFYLVNEPTGSEXLBMVUXMNOERELUVCCMBIRWPJFLKIYQNDVHCDBXBMYVPWGNG
#define PAQKLTINTWOCIQFYLVNEPTGSEXLBMVUXMNOERELUVCCMBIRWPJFLKIYQNDVHCDBXBMYVPWGNG

#include <f/variate_generator/variate_generator.hpp>
#include <f/printer/printer.hpp>

#include <cstddef>
#include <vector>
#include <functional>
#include <cmath>
#include <cassert>
#include <iterator>
#include <iostream>
#include <iomanip>

namespace f
{
    //
    // Interfaces
    //  1) config_unknown_parameters
    //  2) config_merit_function
    // *3) config neighbouring_function
    // *4) config_initialization_function
    //
    template< typename T >
    struct simulated_annealing
    {
        typedef T                                                   value_type;
        typedef value_type*                                         pointer;
        typedef std::size_t                                         size_type;
        typedef std::vector<value_type>                             vector_type;
        typedef std::function<void(pointer)>                        initialization_function_type; 
        typedef std::function<value_type(pointer)>                  function_type; 
        typedef std::function<void(pointer,pointer)>                neighbouring_function_type; 
        typedef std::function<value_type(value_type)>               cooling_function_type;

        size_type                                                   unknown_parameters;
        size_type                                                   equilibrium_steps;
        function_type                                               merit_function;
        value_type                                                  fusion_temperature;
        value_type                                                  frozen_temperature;
        value_type                                                  boltzmann_constant; 
        cooling_function_type                                       cooling_function;

        neighbouring_function_type                                  neighbouring_function;
        initialization_function_type                                initialization_function;

        value_type                                                  elite_residual;
        vector_type                                                 elite_vector;

        void config_unknown_parameters( size_type const unknown_parameters_ )
        {
            unknown_parameters = unknown_parameters_;
        }

        template< typename Function >
        void config_merit_function( Function const& merit_function_ )
        {
            merit_function = merit_function_;
        }

        template< typename Function >
        void config_neighbouring_function( Function const& neighbouring_function_ )
        {
            neighbouring_function = neighbouring_function_;
        }

        template< typename Function >
        void config_initialization_function( Function const& initialization_function_ )
        {
            initialization_function = initialization_function_;
        }

        template< typename Out_Iterator >
        void operator()( Out_Iterator otor )
        {
            parameter_check();
            vector_type x( unknown_parameters, value_type{0} );
            vector_type new_x( unknown_parameters, value_type{0} );

            //default initialization --> 0
            if ( initialization_function )
                initialization_function( x.data() );
            
            //default neighbouring_function -- [-0.1, 0.1]
            if ( !neighbouring_function )
            {
                neighbouring_function = [this]( pointer x, pointer y )
                {
                    variate_generator<value_type> vg( 0.9, 1.1 );
                    size_type const n = (*this).unknown_parameters;
                    for ( size_type i = 0; i != n; ++i )
                    {
                        y[i] = x[i] *vg();
                        if ( std::abs(x[i]) < 0.1 )
                            y[i] += vg() - value_type{1};
                    }
                };
            }

            value_type energy = merit_function( x.data() );

            elite_residual = energy;
            elite_vector = x;

            variate_generator<value_type> vg( 0.0, 1.0 );

            //check boltzmann_constant
            if ( boltzmann_constant < value_type(0) )
            {
                boltzmann_constant = energy / fusion_temperature;
                if ( boltzmann_constant < value_type( 1.0e-10 ) )
                    boltzmann_constant = 0.1;
            }

            std::cerr << "boltzmann_constant = " << boltzmann_constant << "\n";

            value_type current_temperature = fusion_temperature;

            while ( current_temperature > frozen_temperature )
            {
                for ( size_type step_index = 0; step_index != equilibrium_steps; ++step_index )
                {
                    neighbouring_function( x.data(), new_x.data() );
                    value_type const new_energy =  merit_function( new_x.data() );
                    
                    //trace elite
                    if ( new_energy < elite_residual )
                    {
                        elite_residual = new_energy;
                        elite_vector = new_x;
                        std::cerr << "\nnew elite " << elite_residual << "\n";
                    }

                    //if better, update
                    if ( energy > new_energy ) 
                    {
                        x = new_x;
                        energy = new_energy;
                        continue;
                    }

                    //if lucky, update

                    value_type const p =  std::exp( ( energy - new_energy ) / ( boltzmann_constant * current_temperature ) );
                    
                    //std::cout << "\nP - " << p << "\t" << new_energy - energy << "\n";

                    if ( p > vg() )
                    //if ( std::exp( ( energy - new_energy ) / ( boltzmann_constant * current_temperature )  ) > vg() )
                    {
                        x = new_x;
                        energy = new_energy;
                    }
                }
            
                current_temperature = cooling_function( current_temperature );
            }

            std::copy( elite_vector.begin(), elite_vector.end(), otor );

            //check boltzmann_constant
        }

        void operator()()
        {
            std::cout.precision( 15 );
            (*this)( std::ostream_iterator<value_type>( std::cout, "\t" ) );
        }

        void config_equilibrium_steps( size_type const equilibrium_steps_ )
        {
            equilibrium_steps = equilibrium_steps_;
        }

        void config_fusion_temperature( value_type fusion_temperature_ )
        {
            fusion_temperature = fusion_temperature_;
        }

        void config_frozen_temperature( value_type frozen_temperature_ )
        {
            frozen_temperature = frozen_temperature_;
        }

        void config_boltzmann_constant( value_type boltzmann_constant_ )
        {
            boltzmann_constant = boltzmann_constant_;
        }

        template< typename Function >
        void config_cooling_function( Function const& cooling_function_ )
        {
            cooling_function = cooling_function_;
        }

        simulated_annealing() : unknown_parameters(0), equilibrium_steps( 1000 ), fusion_temperature( value_type{1000.0} ), frozen_temperature( value_type{1.0} ), boltzmann_constant( value_type{-1.0} ), cooling_function( [](value_type t) { return t - value_type{1.0}; } )
        {}

        void parameter_check()
        {
            assert( unknown_parameters );
            assert( equilibrium_steps );
            assert( merit_function );
            assert( cooling_function );
            //assert( neighbouring_function );
            //assert( initialization_function );
            assert( cooling_function( fusion_temperature ) < fusion_temperature );
        }
    };//simulated_annealing

}//namespace f

#endif//PAQKLTINTWOCIQFYLVNEPTGSEXLBMVUXMNOERELUVCCMBIRWPJFLKIYQNDVHCDBXBMYVPWGNG

