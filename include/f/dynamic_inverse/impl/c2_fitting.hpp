#ifndef BFJWLXHJOAHQWAEPFDEOSXVVRBPXHSHKFADURJGXMLUFJEBIEOVVSAEPYISPGWDQMUUQPANVX
#define BFJWLXHJOAHQWAEPFDEOSXVVRBPXHSHKFADURJGXMLUFJEBIEOVVSAEPYISPGWDQMUUQPANVX

#include <f/matrix/matrix.hpp>
#include <f/coefficient/coefficient.hpp>
#include <f/optimization/nonlinear/levenberg_marquardt.hpp>
#include <f/dynamic_inverse/impl/ug_initialization.hpp>

#include <iomanip>
#include <iostream>
#include <cstddef>
#include <map>
#include <set>
#include <cassert>
#include <algorithm>
#include <functional>
#include <complex>

namespace f
{

    struct polynomial_info
    {
        typedef std::size_t                            size_type;

        std::map<size_type, unsigned long long>        index_ug_ug_map;
        std::map<unsigned long long, size_type>        ug_ug_index_map;
        size_type                                      square_counter;
        size_type                                      solo_counter;
        size_type                                      double_counter;
        size_type                                      total_term;

        std::function<double( double*, double* )>      target_function;

        std::map<unsigned long long, double>           ug_ug_solution_map;
    };

    static polynomial_info make_polynomial_info( matrix<std::size_t> const& ar, std::size_t column_index = 0 )
    {
        assert( ar.row() == ar.col() );
        assert( column_index < ar.row() );

        polynomial_info info;

        info.square_counter = 0;
        info.solo_counter = 0;
        info.double_counter = 0;

        std::set<unsigned long long> ug_ug_set;

        std::size_t const n = ar.row();

        for ( std::size_t r = 0; r != n; ++r )
        {
            for ( std::size_t c = 0; c != n; ++c )
            {
                std::size_t ug_index_1 = ar[r][c];
                std::size_t ug_index_2 = ar[c][column_index];
                if ( ug_index_1 > ug_index_2 )
                {
                    std::swap( ug_index_1, ug_index_2 );
                }

                if ( 0 == ug_index_2 )
                {
                    continue;    //ignore 0-0 case
                }

                ug_ug_set.insert( ( ug_index_1 << 32 ) + ug_index_2 );
            }
        }

        std::size_t index = 0;
        for ( auto && elem : ug_ug_set )
        {
            ( info.index_ug_ug_map )[index] = elem;
            ( info.ug_ug_index_map )[elem] = index++;

            unsigned long long const ug_index_1 = elem >> 32;
            unsigned long long const ug_index_2 = ( ug_index_1 << 32 ) ^ elem;

            ( ug_index_1 == ug_index_2 ) ? ++info.square_counter : ( ( 0 == ug_index_1 ) ? ++info.solo_counter : ++info.double_counter );
        }

        info.total_term = info.square_counter + info.solo_counter + info.double_counter;

        unsigned long const N = info.total_term;

        //---checked--

        info.target_function = [N] ( double * x, double * a )
        {
            double real_part = x[0];
            double imag_part = x[N + 1];

            for ( unsigned long index = 0; index != N; ++index )
            {
                real_part += x[index + 1] * a[index];
                real_part += x[N + index + 2] * a[index];
            }

            return real_part * real_part + imag_part * imag_part;
        };

        //debug info here
        #if 1
        std::cout << "\n";
        std::cout << "\n-- begin of debug info of make_polynomial_info \n";
        std::cout << "ar is \n" << ar << "\n";
        std::cout << "total term is " << info.total_term << "\n";
        std::cout << "square_counter is " << info.square_counter << "\n";
        std::cout << "solo_counter is " << info.solo_counter << "\n";
        std::cout << "double_counter is " << info.double_counter << "\n";
        std::cout << "the index_ugug map is\n";
        for ( auto && elem : info.index_ug_ug_map )
        {
            std::cout << elem.first << "-" << elem.second << "(" << ( elem.second >> 32 )  << " " << ( ( elem.second << 32 ) >> 32 ) << ")  ";
        }
        std::cout << "\nthe ugug_index map is\n";
        for ( auto && elem : info.ug_ug_index_map )
        {
            std::cout << elem.first << "-" << elem.second << "  ";
        }
        std::cout << "\n-- end of debug info of make_polynomial_info \n\n";
        #endif

        return info;
    }

    static matrix<double> const
    make_polynomial_coef_matrix( matrix<std::size_t> const& ar, matrix<double> const& diag, std::complex<double> const& thickness, std::size_t column_index = 0 )
    {
        assert( ar.row() == ar.col() );
        assert( diag.col() == ar.col() );
        assert( std::abs( std::real( thickness ) ) < 1.0e-10 );
        assert( std::imag( thickness ) > 1.0e-10 );
        assert( column_index < ar.row() );

        unsigned long const n = ar.row();
        unsigned long const total_tilt = diag.row();

        auto const& info = make_polynomial_info( ar, column_index );

        unsigned long new_row = total_tilt * n;
        unsigned long new_col = info.total_term * 2 + 2;

        std::cout << "dim for coef matrix is " << new_row << " by " << new_col << "\n";

        matrix<double> coef_matrix { new_row, new_col, 0.0 };

        for ( std::size_t tilt_index = 0; tilt_index != total_tilt; ++tilt_index )
        {
            coefficient<double> coef { thickness, diag.row_begin( tilt_index ), diag.row_end( tilt_index ) };
            for ( unsigned long r = 0; r != n; ++r )
            {
                unsigned long const r_index = tilt_index * n + r;

                //C2 - coef
                for ( unsigned long c = 0; c != n; ++c )
                {
                    unsigned long long index_1 = ar[r][c];
                    unsigned long long index_2 = ar[c][column_index];
                    if ( index_1 > index_2 ) std::swap( index_1, index_2 );
                    unsigned long long ug_ug =  ( index_1 << 32 ) + index_2;

                    auto const& cf = coef( r, c, column_index );
                    double const real_p = std::real( cf );
                    double const imag_p = std::imag( cf );

                    unsigned long index_real = 0;
                    unsigned long index_imag = info.total_term + 1;

                    auto itor = info.ug_ug_index_map.find(ug_ug);
                    if ( itor != info.ug_ug_index_map.end() )
                    {
                        unsigned long const index = (*itor).second;
                        index_real = index + 1;
                        index_imag = index_real + info.total_term + 1;
                    }
                    else
                    {
                        //std::cerr << "Detected constant term at  (tilt_index, r, c) " << "(" << tilt_index << "," << r << "," << c << ") -- " << real_p << "-" << imag_p << "\n";
                        //std::cerr << "assigning it to [" << r_index << "," << index_real << "] and [" << index_imag << "]\n" ;
                    }

                    coef_matrix[r_index][index_real] += real_p;
                    coef_matrix[r_index][index_imag] += imag_p;

                }//end c loop

                //C1 - coef
                {
                    auto const& c1f = coef( r, column_index );
                    double const real_p = std::real( c1f );
                    double const imag_p = std::imag( c1f );

                    unsigned long const ug_ug = ar[r][column_index];

                    unsigned long index_real = 0;
                    unsigned long index_imag = info.total_term + 1;

                    auto itor = info.ug_ug_index_map.find(ug_ug);

                    if ( itor != info.ug_ug_index_map.end() )
                    {
                        unsigned long const index = (*itor).second;
                        index_real = index + 1;
                        index_imag = index_real + info.total_term + 1;
                    }
                    else
                    {
                        //std::cerr << "Detected constant term at  (tilt_index, r) " << "(" << tilt_index << "," << r << ") -- " << real_p << "-" << imag_p << "\n";
                        //std::cerr << "assigning it to [" << r_index << "," << index_real << "] and [" << index_imag << "]\n" ;
                    }


                    coef_matrix[r_index][index_real] += real_p;
                    coef_matrix[r_index][index_imag] += imag_p;
                }

            }//end r loop

        }//end tilt_index loop

#if 1
        //debug
        std::cout.precision( 3 );

        std::cout << std::setw( 10 ) << std::fixed;

        std::cout << "\n\nthe generated coef matrix is \n\n" << coef_matrix << "\n";
#endif
        return coef_matrix;
    }

    polynomial_info
    solve_c2( matrix<std::size_t> const& ar, matrix<double> const& diag, matrix<double> const& intensity, std::complex<double> const& thickness, std::size_t column_index = 0)
    {
        auto x_mat = make_polynomial_coef_matrix( ar, diag, thickness, column_index );
        auto info = make_polynomial_info( ar, column_index );

        ug_initialization<double> ui;
        ui.config_ar( ar );
        ui.config_intensity_matrix( intensity );
        ui.config_diag_matrix( diag );
        ui.config_thickness( thickness );
        ui.config_column_index( column_index );
        ui.config_ar_dim( ar.row() );
        std::vector<double> c1_guess;
        ui( std::back_inserter( c1_guess ) );

#if 0
        unsigned long new_x_row = x_mat.row() + ar.row() - 1;
        unsigned long new_x_col = x_mat.col();
        matrix<double> new_x{ new_x_row, new_x_col };
        std::copy( x_mat.begin(), x_mat.end(), new_x.begin() );

        unsigned long new_y_row = intensity.size() + ar.row() - 1;
        matrix<double> new_y{ new_y_row, 1 };
        std::copy( intensity.begin(), intensity.end(), new_y.begin() );

        std::size_t new_x_row_index = x_mat.row();
        std::size_t new_y_row_index = intensity.size();

        for ( std::size_t r = 0; r != ar.row(); ++r )
        {
            if ( r == column_index ) continue;

            unsigned long ug_index = ar[r][column_index];
            double c1_approximation = c1_guess[ug_index];
            //update in new_x
            //update in new_y
            double const weigh = 10000.0 + std::accumulate( intensity.col_begin(r), intensity.col_end(r), 0.0 );
            new_x[new_x_row_index][ug_index] = weigh;
            new_y[new_y_row_index][0] = weigh * c1_approximation * weigh * c1_approximation;
            ++new_x_row_index;
            ++new_y_row_index;
        }


        levenberg_marquardt<double> lm;
        lm.config_target_function( info.target_function );
        lm.config_unknown_parameter_size( info.total_term );
        //lm.config_experimental_data_size( x_mat.row() );
        //lm.config_x( x_mat );
        //intensity.reshape( intensity.size(), 1 );
        //lm.config_y( intensity );
        lm.config_experimental_data_size( new_x.row() );
        lm.config_x( new_x );
        lm.config_y( new_y );
        lm.config_eps( 1.0e-10 );
        //lm.config_max_iteration( 1015 );
#endif
#if 1
        matrix<double> new_intensity{ intensity };
        new_intensity.reshape( intensity.size(), 1 );

        //--scaling
        /*   
        for ( std::size_t i = 0; i != new_intensity.row(); ++i )
        {
            double scaler = std::sqrt( 1.0 / new_intensity[i] [0] );
            std::for_each( x_mat.begin(), x_mat.end(), [scaler](double& x){ x *= scaler; } );
        }
        std::fill( new_intensity.begin(), new_intensity.end(), 1.0 );
        */

        //--scaling
        
        levenberg_marquardt<double> lm;
        lm.config_target_function( info.target_function );
        lm.config_unknown_parameter_size( info.total_term );
        lm.config_experimental_data_size( x_mat.row() );
        lm.config_x( x_mat );
        lm.config_y( new_intensity );
        lm.config_eps( 1.0e-10 );
#endif

        std::vector<double> ans;
        if ( lm( std::back_inserter(ans) ) )
            std::cerr << "c2_fitting: Failed to fit the model.\n";


        for ( std::size_t index = 0; index != ans.size(); ++index )
        {
            unsigned long long ugug = info.index_ug_ug_map[index];
            info.ug_ug_solution_map[ugug] = ans[index];
        }

#if 1
        //debug

        std::cout << "\nsolve_c2::solution --> \n";
        for ( auto& elem : info.ug_ug_solution_map )
            std::cout << elem.first << " - " << elem.second << "\n"; 
        std::cout << "\nthe residual is " << lm.chi_square << "\n";
        std::cout << "\nthe total power is " << std::inner_product( intensity.begin(), intensity.end(), intensity.begin(), 0.0 );
        std::cout << "\nthe c1_approximation is\n";
        for ( std::size_t i = 0; i != c1_guess.size(); ++i )
            std::cout << i << "\t\t" << c1_guess[i] << "\n";

        std::cout.precision( 3 );
        std::cout << std::setw( 10 ) << std::fixed;
        //std::cout << "\n\nthe generated coef matrix is \n\n" << new_x << "\n";
        //std::cout << "\n\nthe generated intensity matrix is \n\n" << new_y << "\n";
#endif

        return info;
    }


}//namespace f

#endif//BFJWLXHJOAHQWAEPFDEOSXVVRBPXHSHKFADURJGXMLUFJEBIEOVVSAEPYISPGWDQMUUQPANVX

