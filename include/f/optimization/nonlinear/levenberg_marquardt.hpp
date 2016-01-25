#ifndef ALGEYLQISUBUJIQGBDPWHNEYWFVDQKOENCMHBNKMSURDDVXTIKHTJJHVXDQQJDXFCFCGETIHT
#define ALGEYLQISUBUJIQGBDPWHNEYWFVDQKOENCMHBNKMSURDDVXTIKHTJJHVXDQQJDXFCFCGETIHT

#include <f/matrix/matrix.hpp>
#include <f/matrix/numeric/conjugate_gradient_squared.hpp>
#include <f/derivative/derivative.hpp>
#include <f/matrix/numeric/lu_solver.hpp>

#include <algorithm>
#include <thread>
#include <functional>
#include <cstddef>
#include <cassert>
#include <map>
#include <iostream>
#include <limits>

namespace f
{
    
    template< typename T >
    struct levenberg_marquardt
    {
        typedef T                                                       value_type;
        typedef value_type*                                             pointer;
        typedef std::function<value_type(pointer, pointer)>             target_function_type;     //f(x,a)
        typedef std::function<value_type(pointer)>                      merit_function_type;      //\chi(a)
        typedef std::function<value_type(pointer)>                      partial_function_type;    //\chi(a)
        typedef matrix<value_type>                                      matrix_type;
        typedef std::size_t                                             size_type;
        typedef matrix<partial_function_type>                           partial_function_matrix_type;
        typedef std::map<size_type, target_function_type>               cached_index_jacobian_associate_function_type;

        target_function_type                                            ff;

        matrix_type                                                     x; 
        matrix_type                                                     y; 
        size_type                                                       m;                          //size of parameters to be fixed
        size_type                                                       n;                          //size of experimental data length

        value_type                                                      lambda;
        value_type                                                      lambda_factor;
        value_type                                                      eps;
        matrix_type                                                     va;                         //initial guess

        value_type                                                      chi_square;
        merit_function_type                                             merit_f;

        matrix_type                                                     new_a;
        matrix_type                                                     alpha;
        matrix_type                                                     modified_alpha;
        matrix_type                                                     beta;
        matrix_type                                                     partial_x_a;

        matrix_type                                                     y_;
        matrix_type                                                     y_diff;

        partial_function_matrix_type                                    jacobian_matrix;
        cached_index_jacobian_associate_function_type                   cached_index_jacobian_associate_function;

        size_type                                                       max_iteration;

        template< typename Func >
        void config_target_function( Func const& ff_ )
        {
            ff = ff_;
        }

        void config_x( matrix_type const& x_ )
        {
            x = x_;
        }

        void config_y( matrix_type const& y_ )
        {
            y = y_;
        }

        template< typename P_Function >
        void config_jacobian_matrix( size_type index, P_Function const& pf ) //pf( x, a )
        {
#if 1
            cached_index_jacobian_associate_function[index] = pf;
#endif
#if 0
            assert( index < m );
            assert( jacobian_matrix.row() == n );
            assert( jacobian_matrix.col() == m );
            assert( n );
            assert( m );
            //TODO:
            //      fix here
            //      --store in a set?
            for ( size_type r = 0; r != n; ++r )
                jacobian_matrix[r][index] = [pf, this, r](pointer a){ return pf( &((*this).x[r][0]), a ); };
#endif
        }

        void config_unknown_parameter_size( size_type m_ )
        {
            m = m_;
            va.resize( m, 1 );
            std::fill( va.begin(), va.end(), value_type{0} );
            if ( n )
                jacobian_matrix.resize( n, m );
        }

        void config_experimental_data_size( size_type n_ )
        {
            n = n_;
            if (m)
                jacobian_matrix.resize( n, m );
        }

        void check_initial_status()
        {
            assert( ff );
            assert( x.row() == n );
            assert( y.row() == n );
            assert( y.col() == 1 );
            assert( m );
            assert( n );
            assert( lambda > value_type{0} );
            assert( lambda_factor > value_type{1} );
            assert( eps > value_type{0} );
            assert( va.row() == m );
            assert( va.col() == 1);
        }

        void make_merit_function()
        {
            assert( ff );
            assert( x.row() == n );
            assert( y.row() == n );
            assert( y.col() == 1 );
            merit_f = [this]( pointer a )
            {
                value_type ans{0};
                for ( size_type i = 0; i != (*this).n; ++i )
                {
                    value_type diff = (*this).y[i][0] - ((*this).ff)( &x[i][0], a);
                    ans += diff * diff;
                }
                return ans;
            };
        }

        //update alpha, beta
        int refresh_all_cache()
        {
#if 1
            size_type const max_thread = std::max( size_type{4}, size_type{std::thread::hardware_concurrency()} );

            //update y_
            assert( n );
            y_.resize( n, 1 );

            if (0)
            {
                for ( size_type index = 0; index != n; ++index )
                    y_[index][0] = ff( &x[index][0], &va[0][0] );
            }
            else
            {
                auto const& func = [this]( size_type const index ) { ((*this).y_)[index][0] = ((*this).ff)( &(((*this).x)[index][0]), &(((*this).va)[0][0]) ); };

                auto const& func_batch = [&func]( size_type const index, size_type const jndex ) 
                { 
                    for ( size_type i = index; i != jndex; ++i ) func( i );
                };

                std::vector<std::thread> thread_array;
                size_type const size_per_thread = (n+max_thread-1)/max_thread;

                for ( size_type thread_index = 0; thread_index != max_thread; ++thread_index )
                {   
                    size_type const thread_start_index = thread_index * size_per_thread;
                    size_type const thread_end_index   = std::min( thread_start_index + size_per_thread, n );
                    thread_array.push_back( std::thread{ func_batch, thread_start_index, thread_end_index } );
                }

                for ( size_type thread_index = 0; thread_index != max_thread; ++thread_index )
                    thread_array[thread_index].join();
            }

            //update y_diff
            y_diff = y - y_;

            //update chi_square
            chi_square = std::inner_product( y_diff.begin(), y_diff.end(), y_diff.begin(), value_type{0} );

            /*
            std::cout.precision( 10 );
            std::cout << "\n--LM::refresh_all_cache --> chi_square = " << chi_square << "\n";
            */

            //update partial_x_a
            assert( m );
            if (0)
            {
                partial_x_a.resize( n, m );
            }
            else
            {
                partial_x_a.resize( m, n );
            }

            assert( jacobian_matrix.row() == n );
            assert( jacobian_matrix.col() == m );
            if (0)
            {
                for ( size_type r = 0; r != n; ++r )      //index for x vector
                    for ( size_type c = 0; c != m; ++c )  //index for a vector
                        partial_x_a[r][c] = (jacobian_matrix[r][c])( &va[0][0] );
            }
            else
            {
                auto const& func = [this]( size_type const r, size_type const c )
                {
                    //((*this).partial_x_a)[r][c] = (((*this).jacobian_matrix)[r][c])(&(((*this).va)[0][0]) );
                    ((*this).partial_x_a)[c][r] = (((*this).jacobian_matrix)[r][c])(&(((*this).va)[0][0]) );
                };

                auto const& func_batch = [&func]( size_type const index, size_type const jndex, size_type const second_index ) 
                { 
                    for ( size_type i = index; i != jndex; ++i ) func( i, second_index );
                };

                size_type const size_per_thread = (n+max_thread-1)/max_thread;

                for ( size_type c = 0; c != m; ++c )
                {
                    std::vector<std::thread> thread_array;
                    for ( size_type thread_index = 0; thread_index != max_thread; ++thread_index )
                    {   
                        size_type const thread_start_index = thread_index * size_per_thread;
                        size_type const thread_end_index   = std::min( thread_start_index + size_per_thread, n );
                        thread_array.push_back( std::thread{ func_batch, thread_start_index, thread_end_index, c } );
                    }

                    for ( size_type thread_index = 0; thread_index != max_thread; ++thread_index )
                        thread_array[thread_index].join();
                }

            }

            //update alpha
            alpha.resize( m, m );
            if (0)
            {
                for ( size_type r = 0; r != m; ++r )
                    for ( size_type c = 0; c <=r; ++c )
                    {
                        alpha[r][c] = std::inner_product( partial_x_a.col_begin(r), partial_x_a.col_end(r), partial_x_a.col_begin(c), value_type{0} );
                        alpha[c][r] = alpha[r][c];
                    }
            }
            else
            {
                for ( size_type r = 0; r != m; ++r )
                    for ( size_type c = 0; c <=r; ++c )
                    {
                        alpha[r][c] = std::inner_product( partial_x_a.row_begin(r), partial_x_a.row_end(r), partial_x_a.row_begin(c), value_type{0} );
                        alpha[c][r] = alpha[r][c];
                    }
            }

            //update beta
            beta.resize( m, 1 );
            if (0)
            {
                for ( size_type r = 0; r != m; ++r )
                    beta[r][0] = std::inner_product( y_diff.begin(), y_diff.end(), partial_x_a.col_begin(r), value_type{0} );
            }
            else
            {
                for ( size_type r = 0; r != m; ++r )
                    beta[r][0] = std::inner_product( y_diff.begin(), y_diff.end(), partial_x_a.row_begin(r), value_type{0} );
            }

            return 0;
#endif
#if 0
            //update y_
            assert( n );
            y_.resize( n, 1 );

            for ( size_type index = 0; index != n; ++index )
                y_[index][0] = ff( &x[index][0], &va[0][0] );

            //update y_diff
            y_diff = y - y_;

            //update chi_square
            chi_square = std::inner_product( y_diff.begin(), y_diff.end(), y_diff.begin(), value_type{0} );

            //update partial_x_a
            assert( m );
            partial_x_a.resize( n, m );

            assert( jacobian_matrix.row() == n );
            assert( jacobian_matrix.col() == m );
            for ( size_type r = 0; r != n; ++r )      //index for x vector
                for ( size_type c = 0; c != m; ++c )  //index for a vector
                    partial_x_a[r][c] = (jacobian_matrix[r][c])( &va[0][0] );

            //update alpha
            alpha.resize( m, m );
            for ( size_type r = 0; r != m; ++r )
                for ( size_type c = 0; c <=r; ++c )
                {
                    alpha[r][c] = std::inner_product( partial_x_a.col_begin(r), partial_x_a.col_end(r), partial_x_a.col_begin(c), value_type{0} );
                    alpha[c][r] = alpha[r][c];
                }

            //update beta
            beta.resize( m, 1 );
            for ( size_type r = 0; r != m; ++r )
                beta[r][0] = std::inner_product( y_diff.begin(), y_diff.end(), partial_x_a.col_begin(r), value_type{0} );

            return 0;
#endif
        }

        int solve_new_a()
        {
            assert( alpha.row() == m );
            assert( alpha.col() == m );
            assert( beta.row() == m );
            assert( beta.col() == 1 );

            modified_alpha = alpha;
            //std::for_each( modified_alpha.diag_begin(), modified_alpha.diag_end(), [this]( value_type& v ) { v *= (*this).lambda + value_type{1}; } );
            std::for_each( modified_alpha.diag_begin(), modified_alpha.diag_end(), [this]( value_type& v ) { v += (*this).lambda; } );

            if ( lu_solver( modified_alpha, new_a, beta ) )
                if ( conjugate_gradient_squared( modified_alpha, new_a, beta ) )
            //if ( conjugate_gradient_squared( modified_alpha, new_a, beta ) )
            //    if ( lu_solver( modified_alpha, new_a, beta ) )
                {
                    return 1;
                }

            new_a += va;

            return 0;
        }

        // 0 success
        // 1 failed
        int iterate()
        {
            int const success = 0;
            int const failure = 1;

            if ( refresh_all_cache() ) return failure;

            size_type current_iteration = 0;

            for (;;)
            {
                if ( max_iteration == current_iteration++ ) return success;

                if ( solve_new_a() ) return 1; //should not throw error

                if ( std::isinf(lambda) || std::isnan(lambda) ) return failure;

                value_type const new_chi_square = merit_f( &new_a[0][0] );

                //if ( std::abs(new_chi_square - chi_square) < eps * chi_square || new_chi_square < eps  ) 
                if ( new_chi_square < eps  ) 
                {
                    va = new_a;
                    chi_square = new_chi_square;

                    return success;
                }

                if ( new_chi_square >= chi_square )
                {
                    lambda *= lambda_factor;
                    continue;
                }

                //std::cout << chi_square << " -->> " << new_chi_square << "\n";
                //std::cout << "current solution is \n" << va.transpose() << "\n";

                value_type const opt_ratio = ( chi_square - new_chi_square ) / chi_square;

                lambda /= lambda_factor;
                va = new_a;
                chi_square = new_chi_square;

                if ( opt_ratio < value_type{1.0e-10} ) return success;

                if ( refresh_all_cache() ) return failure;

            }

            assert( !"should never reach here" );

            return 1;
        }

        template<typename Out_Iterator>
        int operator()( Out_Iterator oi ) 
        {
            check_initial_status();

            make_merit_function();

            make_default_jacobian_matrix();

            int ans = iterate();

            //TODO:
            //      last iteration without lambda correction to get correct answer

            std::copy( va.begin(), va.end(), oi );

            return ans;
        }

        levenberg_marquardt()
        {
            lambda = value_type{1.618e-3};
            lambda_factor = value_type{ 1.618 };
            //lambda_factor = value_type{ 6.18 };
            //lambda_factor = value_type{ 2.618 };
            eps = value_type{1.0e-3};
            m = 0;
            n = 0;
            max_iteration = std::numeric_limits<size_type>::max();
        }

        void config_max_iteration( size_type const max_iteration_ )
        {
            max_iteration = max_iteration_;
        }

        void config_lambda( value_type const& lambda_ )
        {
            lambda = lambda_;
        }
        
        void config_lambda_factor( value_type const& lambda_factor_ )
        {
            lambda_factor = lambda_factor_;
        }

        void config_eps( value_type const& eps_ )
        {
            eps = eps_;
        }

        void config_initial_guess( matrix_type const& va_ )
        {
            va = va_;
        }

        void make_default_jacobian_matrix()
        {
#if 0
            assert( jacobian_matrix.row() == n );
            assert( jacobian_matrix.col() == m );
            for ( size_type r = 0; r != n; ++r )
            {
                auto const& fxa = [r, this]( pointer a ) { return ff( &((*this).x[r][0]), a ); };
                for ( size_type c = 0; c != m; ++c )  //index for a vector
                {
                    if ( !jacobian_matrix[r][c] ) //make default ones
                        jacobian_matrix[r][c] = make_derivative( fxa, c );
                }
            }
#endif
#if 1
            assert( m );
            assert( n );

            jacobian_matrix.resize( n, m );

            for ( size_type r = 0; r != n; ++r )
            {
                pointer px = &(x[r][0]);
                auto const& fxa = [this,px]( pointer a ) { return ((*this).ff)( px, a ); };
                for ( size_type c = 0; c != m; ++c )
                {
                    if ( cached_index_jacobian_associate_function.find(c) == cached_index_jacobian_associate_function.end() ) 
                        jacobian_matrix[r][c] = make_derivative(fxa, c); 
                    else
                        jacobian_matrix[r][c] = [this,c,px](pointer a) {return (((*this).cached_index_jacobian_associate_function)[c])(px, a); };
#if 0
                    jacobian_matrix[r][c] = cached_index_jacobian_associate_function.find(c) == cached_index_jacobian_associate_function.end() ? 
                                            partial_function_type{ make_derivative(fxa, c) } : 
                                            partial_function_type{ [this,c,px](pointer a) {return (((*this).cached_index_jacobian_associate_function)[c])(px, a); } };
#endif
                }
            }
#endif
        }
    
    };

}//namespace f

#endif//ALGEYLQISUBUJIQGBDPWHNEYWFVDQKOENCMHBNKMSURDDVXTIKHTJJHVXDQQJDXFCFCGETIHT

