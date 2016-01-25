#ifndef RGHVGWIMFHIPGCOAHHGPQDCVPFCVAXMYJAXDOPTNVCVCCPKTPIJWHMRRVGWCIRSQKFTEYXPNQ
#define RGHVGWIMFHIPGCOAHHGPQDCVPFCVAXMYJAXDOPTNVCVCCPKTPIJWHMRRVGWCIRSQKFTEYXPNQ

#include <f/algorithm/for_each.hpp>
#include <f/matrix/matrix.hpp>
#include <f/derivative/derivative.hpp>

#include <algorithm>
#include <functional>
#include <cstddef>

namespace f
{

    template< typename T, typename Concret_Descent >
    struct rprop_base
    {
        typedef T                                               value_type;
        typedef Concret_Descent                                 zen_type;
        typedef value_type*                                     pointer;
        typedef std::function<value_type(pointer)>              function_type;
        typedef std::size_t                                     size_type;

        function_type                                           merit_function;

        size_type                                               variable_length;
        size_type                                               total_step;
        size_type                                               current_step;

        matrix<value_type>                                      initial_guess;
        matrix<value_type>                                      direction;
        matrix<value_type>                                      step_size;
        matrix<value_type>                                      delta_solution;
        matrix<value_type>                                      current_solution;

        matrix<int>                                             gradient_k_sign;    //gradient at step k
        matrix<int>                                             gradient_k_1_sign;  //gradient at step k-1

        value_type                                              eps;
        value_type                                              current_residual;

        value_type                                              delta_max;
        value_type                                              delta_min;
        value_type                                              eta_plus;
        value_type                                              eta_minus;

        std::function<void(pointer)>                            iteration_over_function;
        
        void impl_step_size() 
        { 
            for_each( gradient_k_sign.begin(), gradient_k_sign.end(), gradient_k_1_sign.begin(), step_size.begin(), delta_solution.begin(),
                      []( int& k, int k_1, value_type& s, value_type d_ )
                      {
                        if ( k * k_1 > 0 ) 
                            s = std::min( delta_max, std::abs(d_) * eta_plus );  
                        else if ( k * k_1 < 0 )
                        {
                            s = std::max( delta_min, std::abs(d_) * eta_minus );  
                            k = 0;
                        }
                        else
                            s = std::abs( d_ );
                      } 
                    );
        }

        void impl_direction() 
        { 
            for_each( gradient_k_sign.begin(), gradient_k_sign.end(), gradient_k_1_sign.begin(), direction.begin(),
                      []( int k, int k_1, value_type& d )
                      {
                        if ( k * k_1 >= 0 ) d = ( k < 0 ) ? 1.0 : -1.0; 
                        else d = -1.0;
                      } 
                    );
        }

        void iterate()
        {
            auto& zen = static_cast<zen_type&>(*this);

            update_gradient();

            zen.make_direction();
            zen.make_step_size();

            for_each( step_size.begin(), step_size.end(), direction.begin(), delta_solution.begin(), []( value_type const s_, value_type const d_, value_type& d__ ){ d__ = d_ * s_; } );
            current_solution += delta_solution;

            gradient_k_1_sign.swap(gradient_k_sign);

            iteration_over_function( current_solution.data() );
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

        void make_step_size() //default
        {
            auto& zen = static_cast<zen_type&>(*this);
            zen.impl_step_size();
        }

        void make_direction() //default
        {
            auto& zen = static_cast<zen_type&>(*this);
            return zen.impl_direction();
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

        void config_total_steps( size_type loop_ )
        {
            total_step = loop_;
        }

        void config_eps( value_type const eps_ )
        {
            eps = eps_;
        }

        template< typename F >
        void config_iteration_function( F func )
        {
            iteration_over_function = func;
        }

        template< typename Function >
        rprop( Function const& func_, size_type len_, size_type loop_ = 1000 ) : merit_function{ func_ }, variable_length{ len_ }, total_step{ loop_ }, 
            current_step{ 0 }, eps{ 1.0e-10 }, delta_max{ 0.1 }, delta_min{ 0.0 }, eta_plus{ 1.2 }, eta_minus{ 0.5 }, iteration_over_function{ [](value_type,pointer){} }
        {
            initial_guess.resize( variable_length, 1 );
            direction.resize( variable_length, 1 );
            step_size.resize( variable_length, 1 );
            delta_solution.resize( variable_length, 1 );
            current_solution.resize( variable_length, 1 );
            gradient_k_sign.resize( variable_length, 1 );
            gradient_k_1_sign.resize( variable_length, 1 );

            std::fill( gradient_k_sign.begin(), gradient_k_sign.end(), 0 );
            std::fill( gradient_k_1_sign.begin(), gradient_k_1_sign.end(), 0 );
            std::fill( initial_guess.begin(), initial_guess.end(), value_type{0} );
            std::fill( current_solution.begin(), current_solution.end(), value_type{0} );
            std::fill( delta_solution.begin(), delta_solution.end(), value_type{0.00125} );
        }

        void update_gradient()
        {
            value_type const central = merit_function( current_solution.data() );
            value_type const delta = 1.0e-8;
            for ( size_type index = 0; index != variable_length; ++index )
            {
                current_solution[index][0] += delta;

                value_type const rhs = merit_function( current_solution.data() );

                gradient_k_1_sign[index][0] = ( rhs > central ) ? 1 : ( ( rhs < central ) ? -1 : 0 );

                current_solution[index][0] -= delta;
            }
        }

    };//struct rprop


    template< typename T >
    struct rprop_plus : rprop_base< T, rprop_plus<T> >
    {
        rprop_plus( auto ... args ) : rprop_base< T, rprop_plus<T> >{ args... } {}
    };

    template< typename T >
    struct rprop_minus : rprop_base< T, rprop_minus<T> >
    {
        rprop_minus( auto ... args ) : rprop_base< T, rprop_minus<T> >{ args... } {}

        void impl_direction() 
        { 
            for_each( gradient_k_sign.begin(), gradient_k_sign.end(), gradient_k_1_sign.begin(), direction.begin(),
                      []( int k, int k_1, value_type& d )
                      {
                        d = ( k < 0 ) ? 1.0 : -1.0; 
                      } 
                    );
        }
    };

    template< typename T >
    struct irprop_plus : rprop_base< T, irprop_plus<T> >
    {
        irprop_plus( auto ... args ) : rprop_base< T, irprop_plus<T> >{ args... } {}

        void impl_step_size() 
        { 
            for_each( gradient_k_sign.begin(), gradient_k_sign.end(), gradient_k_1_sign.begin(), step_size.begin(), delta_solution.begin(),
                      []( int& k, int k_1, value_type& s, value_type d_ )
                      {
                        if ( k * k_1 > 0 ) 
                            s = std::min( delta_max, std::abs(d_) * eta_plus );  
                        else if ( k * k_1 < 0 )
                        {
                            s = std::max( delta_min, std::abs(d_) * eta_minus );  

                            //if E(t) > E(t-1) ... s = std::abs(d_);:wq

                            k = 0;
                        }
                        else
                            s = std::abs( d_ );
                      } 
                    );
        }

    };


}//namespace f

#endif//RGHVGWIMFHIPGCOAHHGPQDCVPFCVAXMYJAXDOPTNVCVCCPKTPIJWHMRRVGWCIRSQKFTEYXPNQ

