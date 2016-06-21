#ifndef RHOKBLJCUSWTSNUQCTUBRKYAHKVQCLUIYBQMESOABGQXQBJXKUUGIXMJHNTOHQNUQPDUVTHDJ
#define RHOKBLJCUSWTSNUQCTUBRKYAHKVQCLUIYBQMESOABGQXQBJXKUUGIXMJHNTOHQNUQPDUVTHDJ

#include <f/matrix/matrix.hpp>
#include <f/optimization/nonlinear/levenberg_marquardt.hpp>

#include <iterator>
#include <algorithm>
#include <vector>
#include <cstddef>
#include <cassert>
#include <numeric>
#include <ctime>
#include <cstdlib>

namespace f
{

    template< typename T = double >
    struct double_square_modeling
    {
        typedef std::size_t                 size_type;
        typedef T                           value_type;
        typedef std::vector<value_type>     array_type;
        typedef matrix<value_type>          matrix_type;

        size_type                           m_;//ug size
        array_type                          y_;
        array_type                          x_real_; //2D -> 1D
        array_type                          x_imag_; //2D -> 1D

        array_type                          a_;
        value_type                          target_residual_threshold_;
        value_type                          residual_;
        int                                 fitting_flag_; // 0 for success, 1 for failure or not being fitted

        size_type                           max_iteration_;

        void set_max_iteration( size_type const max_iteration )
        {
            max_iteration_ = max_iteration;
        }

        void set_target_residual_threshold( value_type target_residual_threshold )
        {
            target_residual_threshold_ = target_residual_threshold;
        }

        double_square_modeling() : target_residual_threshold_( value_type{1.0e-15} ), fitting_flag_( 1 ), max_iteration_( 1 ) {}

        void fit()
        {
            assert( y_.size() >= m_ );

            //
            // m is the [U_g] size, numbering from 0 to m-1
            // n is the intensity measurement times, which is supposed to be [tilt_size X Ar_dim]
            //
            size_type const m = m_;
            size_type const n = x_real_.size() / m;

            //
            // expand data to matrice
            //
            matrix_type intensity{y_.size(), 1};
            std::copy( y_.begin(), y_.end(), intensity.col_begin(0) );

            matrix_type x_lhs{ x_real_.size() / m, m };
            std::copy( x_real_.begin(), x_real_.end(), x_lhs.begin() );

            matrix_type x_rhs{ x_imag_.size() / m, m };
            std::copy( x_imag_.begin(), x_imag_.end(), x_rhs.begin() );

            matrix_type coef = x_lhs ||  x_rhs;

            //std::cout << "\ny array is \n" << intensity << "\n";
            //std::cout << "\nx is \n" << coef << "\n";

            //
            // the model, where [a] is parameters to be fitted
            //
            auto const& fxa = [m]( value_type* x, value_type* a )
            {
                value_type const a_0 = a[0];
                value_type const x_0 = x[0];
                value_type const x_m = x[m];
                value_type const real = std::inner_product( x+1, x+m, a+1, x_0 );
                value_type const imag = std::inner_product( x+m+1, x+m+m, a+1, x_m );
                return a_0 * ( real*real + imag*imag );
            };

            //
            // configure levenberg_marquardt algorithm 
            //
            levenberg_marquardt<value_type> lm;
            lm.config_target_function( fxa );
            lm.config_unknown_parameter_size( m );
            lm.config_experimental_data_size( n );
            lm.config_x( coef );
            lm.config_y( intensity );
            lm.config_eps( value_type{1.0e-20} );
            lm.config_max_iteration( 100 );

            //config_initial_guess
            lm.config_jacobian_matrix( 0,    [m]( value_type* x, value_type* a )
                                                {
                                                    value_type const x_0 = x[0];
                                                    value_type const x_m = x[m];
                                                    value_type const real = std::inner_product( x+1, x+m, a+1, x_0 );
                                                    value_type const imag = std::inner_product( x+m+1, x+m+m, a+1, x_m );
                                                    return real*real + imag*imag;
                                                } 
                                       );

            for ( size_type index = 1; index != m; ++index )
                lm.config_jacobian_matrix( index, [m, index]( value_type* x, value_type* a )
                                                  {
                                                      value_type const a_0   = a[0];
                                                      value_type const x_0   = x[0];
                                                      value_type const x_m   = x[m];
                                                      value_type const x_i   = x[index];
                                                      value_type const x_m_i = x[index+m];
                                                      value_type const real  = std::inner_product( x+1, x+m, a+1, x_0 );
                                                      value_type const imag  = std::inner_product( x+m+1, x+m+m, a+1, x_m );
                                                      return ( a_0 + a_0 ) * ( x_i * real + x_m_i * imag );
                                                  }
                                        );

            matrix_type ug{ m, 1 };

            if ( a_.size() != m )
            {
                a_.resize( m );
                std::copy( y_.begin(), y_.begin()+m, a_.begin() );
                //std::fill( a_.begin(), a_.end(), 0.1 );
                std::fill( a_.begin(), a_.end(), 0.0 );
            }

            std::copy( a_.begin(), a_.end(), ug.begin() );
#if 0
            //
            // execute levenberg_marquardt 
            //
            std::srand( unsigned ( std::time(0 ) ) );
            size_type iterations_so_far = 0;

            //TODO:
            //      here should return an elite one

            do
            {

                std::random_shuffle( ug.begin()+1, ug.end() );
                lm.config_initial_guess( ug );
                fitting_flag_ = lm( ug.col_begin(0) );
                residual_ = lm.chi_square;
                if ( residual_ > target_residual_threshold_ )
                    fitting_flag_ = 1;

                if ( iterations_so_far++ == max_iteration_ )
                    break;
            }
            while ( fitting_flag_ );
#endif
            lm.config_initial_guess( ug );
            fitting_flag_ = lm( ug.col_begin(0) );
            residual_ = lm.chi_square;

            //
            // restore fitted parameters
            //
            a_.resize( m );
            std::copy( ug.begin(), ug.end(), a_.begin() );
        }

        void set_parameter_size( size_type m )
        {
            m_ = m;
        }

        //need optmization with only one array x
        void register_entry( value_type y, array_type const& x_real, array_type const& x_imag )
        {
            assert( y >= value_type{0} );
            assert( x_real.size() == m_ );
            assert( x_imag.size() == m_ );

            y_.push_back( y );
            for ( size_type index = 0; index != m_; ++index )
            {
                x_real_.push_back( x_real[index] );
                x_imag_.push_back( x_imag[index] );
            }
        }

        template< typename Input_Iterator_1, typename Input_Iterator_2 >
        void register_entry( value_type y, Input_Iterator_1 it_1, Input_Iterator_2 it_2 )
        {
            assert( y >= value_type{0} );
            y_.push_back( y );
            for ( size_type index = 0; index != m_; ++index )
            {
                x_real_.push_back( *it_1++ );
                x_imag_.push_back( *it_2++ );
            }

            //set the fitting flag to [not being fitted]
            fitting_flag_ = 1;
        }

        template< typename Output_Iterator >
        void output( Output_Iterator oi )
        {
            std::copy( a_.begin(), a_.end(), oi );
        }

        value_type residual()
        {
            return residual_;
        }

        template< typename Input_Iterator >
        void guess( Input_Iterator begin, Input_Iterator end )
        {
            a_.clear();
            std::copy( begin, end, std::back_inserter( a_ ) );
        }
    };

}//namespace f

#endif//RHOKBLJCUSWTSNUQCTUBRKYAHKVQCLUIYBQMESOABGQXQBJXKUUGIXMJHNTOHQNUQPDUVTHDJ

