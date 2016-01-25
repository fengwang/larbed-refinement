#ifndef MSECOND_ORDER_APPROXIMATION_HPP_INCLUDED_FDSIOUHNSADFLKJHNSADFLKJ34OIUHA
#define MSECOND_ORDER_APPROXIMATION_HPP_INCLUDED_FDSIOUHNSADFLKJHNSADFLKJ34OIUHA

#include <f/pattern/elementary_pattern.hpp>
#include <f/pattern/coefficient_composer.hpp>
#include <f/test/test.hpp>
#include <f/matrix/matrix.hpp>
#include <f/singleton/singleton.hpp>
#include <f/pattern/double_square_solver.hpp>
#include <f/coefficient/expm.hpp>

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <vector>
#include <list>

#include <boost/lexical_cast.hpp>

namespace f
{

    namespace second_order_approximation_private
    {
        struct patterns
        {
            typedef matrix<double>             matrix_type;
            typedef matrix<unsigned long>      u_matrix_type;
            typedef std::complex<double>       complex_type;
            typedef matrix<complex_type>        complex_matrix_type;

            matrix_type diags;
            u_matrix_type ar;
            complex_type ipit;

            patterns( std::string const& path = std::string {"data/SSTO/"}, double pit = 5.0 )
            {
                //load ar
                ar.load( path + std::string {"Ar.txt"} );
                assert( ar.row() == ar.col() );
                unsigned long const row = ar.row();
                unsigned long const tilt_number = 197;
                // load diag
                diags.resize( row, tilt_number );
                matrix_type tmp_diag { row, 1 };
                for ( unsigned long c = 0; c != tilt_number; ++c )
                {
                    std::string const num = boost::lexical_cast<std::string>( c );
                    std::string const file_name = path + std::string {"Diag_"} + num + std::string {".txt"};
                    tmp_diag.load( file_name );
                    std::copy( tmp_diag.begin(), tmp_diag.end(), diags.col_begin( c ) );
                }
                //load ug
                matrix_type ug_raw;
                ug_raw.load( path + "ug.txt" );
                matrix_type& ug = singleton<matrix_type>::instance();
                ug.resize( ug_raw.row(), 1 );
                std::copy( ug_raw.col_begin( 0 ), ug_raw.col_end( 0 ), ug.col_begin( 0 ) );
                ipit = complex_type { 0.0, pit };
            }

            matrix_type const make_intensity( matrix_type const& new_ug, unsigned long column_index )
            {
                complex_matrix_type     A( ar.row(), ar.col() );

                for ( unsigned long r = 0; r != ar.row(); ++r )
                {
                    for ( unsigned long c = 0; c != ar.col(); ++c )
                        A[r][c] = complex_type{new_ug[ar[r][c]][0], 0.0};
                    A[r][r] = complex_type{0.0, 0.0};
                }
matrix_type ans( diags.row(), diags.col(), double{0.0} );

                for ( unsigned long c = 0; c != diags.col(); ++c )
                {
                    std::transform( diags.col_begin(c), diags.col_end(c), A.diag_begin(), [](double d){ return complex_type{d, 0.0}; } );
                    for ( unsigned long r = 0; r != column_index; ++r )
                        ans[r][c] = std::norm(expm_2(A, ipit, r, column_index));
                }

                return ans;
            }

            double pattern_residual( matrix_type const& new_ug, unsigned long column_index )
            {
                matrix_type& ug = singleton<matrix_type>::instance();
                auto const& i_sim2 = make_intensity( ug, column_index );
                auto const& i_gen = make_intensity( new_ug, column_index );
                auto const& diff = i_sim2 - i_gen;
                double const nm = std::inner_product( diff.begin(), diff.end(), diff.begin(), double{0.0} );
                return nm;
            }

            void dump()
            {
                std::cout << "dumping the patterns.\n";
                std::cout << "The Ar is \n" << ar << "\n";
                std::cout << "The Diag is \n" << diags << "\n";
                std::cout << "The ipit is \n" << ipit << "\n";
            }

        };//struct patterns

    }//namespace second_order_approximation_private

    struct second_order_approximation
    {
        typedef unsigned long                           size_type;
        typedef matrix<double>                          matrix_type;
        typedef std::vector<matrix_type>                matrix_array_type;
        typedef std::list<matrix_type>                  matrix_list_type;

        second_order_approximation_private::patterns    p;
        size_type                                       column_index;
        size_type                                       total_ug;
        size_type                                       total_tilt;
        matrix_array_type                               v_fit;

        matrix_type                                     abs_mat;//this matrix stores all the absolute value of ugs,  i.e., |U_{x}| -- abs_mat[x][0]
        matrix_list_type                                solution_list;

        second_order_approximation( std::string const& path_ = std::string {"data/SSTO/"}, double pit_ = 5.0, unsigned long column_index_ = 0, unsigned long total_ug_ = 156, unsigned long total_tilt_ = 45 )
            : p( path_, pit_ ), column_index( column_index_ ), total_ug( total_ug_ ), total_tilt( total_tilt_ ), abs_mat( total_ug_, 1 )
        {
            for ( unsigned long index = 0; index != total_tilt; ++ index )
            {
                if ( index == column_index ) continue;
                fit_pattern( index );
            }

            extract_absoluate_value();
            make_solution_list();
        }

        void dump()
        {
            //for ( auto const& mat : v_fit )
            //    std::cout << mat << "\n";

            //std::cout << "\n\nthe abs_mat is \n" << abs_mat << "\n";

            std::cout << "\nthe solution list is \n";
            for ( auto const& sol : solution_list )
            {
                std::cout.precision(15);
                std::cout << sol << "\n\n\n\n";
            }

            /*
            auto const& first = *(solution_list.begin());
            auto const& last = *(solution_list.rbegin());
            for ( size_type r = 0; r != first.row(); ++r )
            {
                if ( std::abs( first[r][0] - last[r][0] ) > 1.0e-5 )
                {
                    std::cerr << "\nthe " << r << "th element is different----    " << first[r][0] << "\t" << last[r][0] << "=" << std::abs( first[r][0] - last[r][0] );
                }
            }
            */

            unsigned long counter = 0;
            matrix_type new_ug;
            for ( auto const& mat : solution_list )
            {
                new_ug.resize( mat.row(), 1 );
                std::copy( mat.col_begin(0), mat.col_end(0), new_ug.col_begin(0) );
                std::cerr << "\nfor solution " << counter++ << ", the residual is " << p.pattern_residual( new_ug, column_index ) << "\n";
            }

        }

        void make_solution_list()
        {
            solution_list.push_back( v_fit[0] );
            solution_list.push_back( -v_fit[0] );

            matrix_type result_state;

            for ( size_type index = 1; index != v_fit.size(); ++index )
            {
                matrix_type const& new_state = v_fit[index];
                matrix_list_type new_solution_list;
                for ( auto const& old_state : solution_list )
                {
                    if ( 0 == can_merge_new_state( new_state, old_state, result_state ) )
                        new_solution_list.push_back( result_state );
                    if ( 0 == can_merge_new_state( -new_state, old_state, result_state ) )
                        new_solution_list.push_back( result_state );
                }
                new_solution_list.swap( solution_list );
            }
        }

        //if possible, generate result state, return 0;
        //otherwise, return 1;
        //TODO:
        // test
        int can_merge_new_state( matrix_type const& new_state, matrix_type const& old_state, matrix_type& result_state ) const
        {
            assert( new_state.row() == old_state.row() );
            assert( new_state.col() == old_state.col() );
            double const eps = 1.0e-20;

            auto const& is_valid = []( double const v ) { double const abs_v = std::abs(v); if ( abs_v > 1.0e-20 && abs_v < 0.1 ) return true; return false; };

            bool merge_flag = true;
            for ( size_type r = 0; r != new_state.row(); ++r )
            {
                double const new_rr = std::abs( new_state[r][r] );
                if ( new_rr < 0.1 && new_rr > eps && new_state[r][r] < 0.0 )
                {
                    merge_flag = false;
                    break;
                }
                for ( size_type c = 0; c != new_state.col(); ++c )
                {
                    double const new_rc = std::abs( new_state[r][c] );
                    double const old_rc = std::abs( old_state[r][c] );
                    if ( new_rc < 0.1 && new_rc > eps && old_rc < 0.1 && old_rc > eps )
                    {
                        if ( new_state[r][c] * old_state[r][c] < eps ) //not of same sign
                        {
                            merge_flag = false;
                            break;
                        }
                    }
                }
                if ( !merge_flag ) break; //jump out loop
            }

            if ( merge_flag )
            {
                result_state = old_state;
                for ( size_type r = 0; r != new_state.row(); ++r )
                    for ( size_type c = 0; c != new_state.col(); ++c )
                    {
                        //TODO:
                        // fix here with weigh
                        double const abs_rc = std::abs( new_state[r][c] );
                        if ( abs_rc > eps && abs_rc < 0.1 )
                        {
                            result_state[r][c] = new_state[r][c];
                            result_state[c][r] = result_state[r][c];
                        }
                    }

                size_type loop_merge = 4;

                while( loop_merge-- )
                {

                    //post process result_state
                    for ( size_type r = 0; r != result_state.row(); ++r )
                        for ( size_type c = 0; c != result_state.col(); ++c )
                        {
                            double const row_r = std::abs( result_state[r][0] ); //the first column
                            double const col_c = std::abs( result_state[0][c] ); //the first row
                            //TODO:
                            //fix with weigh
                            if ( row_r > eps && row_r < 0.1 && col_c > eps && col_c < 0.1 )
                            {
                                result_state[r][c] = result_state[r][0] * result_state[0][c];
                                result_state[c][r] = result_state[r][c];
                            }
                        }

                    //TODO:
                    // fix with weigh
                    for ( size_type r = 1; r != result_state.row(); ++r )
                    {
                        if ( !is_valid( result_state[r][0] ) ) continue;
                        for ( size_type c = 1; c != result_state.col(); ++c )
                        {
                            if ( !is_valid( result_state[r][c] ) ) continue;

                            if ( is_valid( result_state[0][c] ) ) continue;

                            result_state[0][c] = result_state[r][c] / result_state[r][0];
                            result_state[c][0] = result_state[0][c];
                        }
                    }
                }


                return 0;
            }

            return -1;
        }

        void extract_absoluate_value()
        {
            std::fill( abs_mat.begin(), abs_mat.end(), 1.0 );
            abs_mat[0][0] = 0.0; //ug_0 is always zero
            //for each matrix in v_fit, gather element in the first row/col, which is Ux
            for ( auto const& mat : v_fit )
            {
                //scan the first row, only one element is useful
                for ( size_type r = 0; r != mat.row(); ++r )
                {
                    auto const abs_v = std::abs( mat[r][0] );
                    if ( abs_v < 0.1 && abs_v > 1.0e-10 )
                    {
                        abs_mat[r][0] = abs_v;
                        break;
                    }
                }
            }

            //have some abs value not computed
            while ( std::count_if( abs_mat.begin()+1, abs_mat.end(), []( double d ) { if ( d > 0.1 ) return true; return false; } ) != 0 )
            {
                //loop in every fitting matrix
                for ( auto const& mat : v_fit )
                {
                    //for each element in the fitting matrix ---- do I neet to change the data structure?
                    //TODO:
                    //      should be weigh based
                    for ( size_type r = 1; r != mat.row(); ++r )
                        for ( size_type c = 1; c != mat.col(); ++c )
                        {
                            double const mrc  = std::abs( mat[r][c] );
                            if ( mrc < 0.1 )
                            {
                               if ( abs_mat[r][0] < 0.1 && abs_mat[c][0] > 0.1 )
                                   abs_mat[c][0] = mrc / abs_mat[r][0];
                               if ( abs_mat[r][0] > 0.1 && abs_mat[c][0] < 0.1 )
                                   abs_mat[r][0] = mrc / abs_mat[c][0];
                            }
                        }
                }
            }
        }

        void fit_pattern( unsigned long index )
        {
            matrix<double> diag {p.diags.row(), 1};
            coefficient_composer cc;
            cc.reset();
            double total_power = 0.0;
            double u_current = 0.0;
            for ( unsigned long c = 0; c != p.diags.col(); ++c  )
            {
                std::copy( p.diags.col_begin( c ), p.diags.col_end( c ), diag.begin() );
                elementary_pattern ep { column_index, p.ipit, p.ar, diag };
                ep.make_i_c2();
                auto const real_p = ep.make_simple_real_s_c2_polynomial( index, column_index );
                auto const imag_p = ep.make_simple_imag_s_c2_polynomial( index, column_index );
                auto const real_v = eval( real_p );
                auto const imag_v = eval( imag_p );
                auto const power = real_v * real_v + imag_v * imag_v;
                total_power += power;
                cc.register_real_imag_intensity( real_p, imag_p, power );
                //guess the first Ug's norm
                coefficient<double> const coef( p.ipit, diag.begin(), diag.end() );
                auto const& c1 = coef( index, column_index );
                double const c1_norm = std::norm( c1 );
                double const guess = std::sqrt( power / c1_norm );
                u_current += guess;
            }
            cc.process();
            //loop here
            unsigned long const max_attempt = 5;
            bool first_attempt = true;
            matrix<double> fit_result;
            //guess the first one in fit_result
            fit_result.resize( cc.vx.col() >> 1, 1 );
            for ( unsigned long order = 0; order != max_attempt; ++order )
            {
                fit_result[0][0] = u_current / p.diags.col();
                if ( first_attempt )
                {
                    first_attempt = false;
                }
                else
                {
                    //random gen
                    std::copy( cc.vx.begin(), cc.vx.begin() + fit_result.size() - 1, fit_result.begin() + 1 );
                    std::random_shuffle( fit_result.begin() + 1, fit_result.end() );
                }
                double_square_solver( cc.vx, cc.vy, fit_result );
                double residual = 0.0;
                for ( unsigned long r = 0; r != cc.vx.row(); ++r )
                {
                    unsigned long  const n = fit_result.size();
                    double const real_v = std::inner_product( fit_result.begin(), fit_result.end(), cc.vx.row_begin( r ), 0.0 );
                    double const imag_v = std::inner_product( fit_result.begin(), fit_result.end(), cc.vx.row_begin( r ) + n, 0.0 );
                    double const diff = real_v * real_v + imag_v * imag_v - cc.vy[r][0];
                    residual += std::abs( diff );
                }
                bool successful = true;
                for ( unsigned long id = 1; id != fit_result.row(); ++id )
                {
                    if ( std::abs( fit_result[id][0] ) > 6.0e-3 )
                    {
                        successful = false;
                        break;
                    }
                }
                if ( std::abs( residual - total_power ) / total_power < 1.0e-5 )
                    successful = false;

                if ( !successful )
                {
                    std::cerr << "The fitting for index " << index << " failed.\n";
                    //continue;
                }
                else
                {
                    std::cerr << "The fitting for index " << index << " is successful.\n";
                }
                if ( successful )
                    break;
            }

            index_maker& im = singleton<index_maker>::instance();
            matrix<double> fit( total_ug + 1, total_ug + 1, 1.0 );
            for ( auto const& element : im.key_index )
            {
                unsigned long const key = element.first;
                unsigned long const key1 = key >> 16;
                unsigned long const key2 = ( key1 << 16 ) ^ key;
                fit[key1][key2] = fit_result[element.second][0];
                fit[key2][key1] = fit_result[element.second][0];
            }
            v_fit.push_back( fit );
            im.reset();
        }

    };//struct second_order_approximation

}//namespace f

#endif//_SECOND_ORDER_APPROXIMATION_HPP_INCLUDED_FDSIOUHNSADFLKJHNSADFLKJ34OIUHA

