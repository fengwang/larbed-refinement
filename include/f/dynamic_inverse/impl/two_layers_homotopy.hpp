#ifndef FYNRKJWVBDGMMXDTKELPDSTMWMLSBJGKRFLTWRFNRYWHURBXEPDMAUUTHIGPLEPNLGOFCKSLY
#define FYNRKJWVBDGMMXDTKELPDSTMWMLSBJGKRFLTWRFNRYWHURBXEPDMAUUTHIGPLEPNLGOFCKSLY

#include <f/dynamic_inverse/impl/c1_matrix.hpp>
#include <f/dynamic_inverse/impl/double_square_coef_matrix.hpp>
#include <f/dynamic_inverse/impl/double_square_ug_fitting.hpp>
#include <f/matrix/matrix.hpp>
#include <f/matrix/numeric/expm.hpp>

#include <complex>
#include <cassert>
#include <cstddef>
#include <algorithm>
#include <vector>

namespace f
{

    namespace homotopy_details
    {
#if 0
        | C1_00 C1_01 ...... C1_0n | | expm 0 |   | I_0 |
        | C1_10 C1_11 ...... C1_1n | | expm 1 |   | I_1 |
        | ........................ | | ...... | = | ... |
        | ........................ | | ...... |   | ... |
        | C1_n0 C1_n1 ...... C1_nn | | expm n |   | I_n |
#endif
    //for one tilt  ar matrix only
    template< typename T = double >
    T make_two_layers_homotopy( matrix<std::size_t> const& ar,                      // beam index matrix
                                matrix<T> const& diag,                              // row major
                                matrix<T> const& intensity,                         // row major
                                matrix<T> const& ug,                                // column major
                                std::complex<T> const& lhs_thickness,               // C1 approximation part
                                std::complex<T> const& rhs_thickness,               // expm part
                                matrix<T>& new_ug,                                  // fitting result
                                std::size_t column_index = 0 )                      // 
    {
        //dimension concept check
        assert( ar.row() );
        assert( ar.row() == ar.col() );
        assert( ar.row() == diag.col() );
        assert( diag.row() == intensity.row() );
        assert( diag.row() );
        assert( diag.col() == intensity.col() );
        assert( ug.col() == 1 );
        assert( ug.size() > *std::max_element( ar.begin(), ar.end() ) );
        assert( std::imag( lhs_thickness) > T{0} );
        assert( std::imag( rhs_thickness) > T{0} );
        assert( std::abs(std::real( lhs_thickness)) < T{1.0e-10} );
        assert( std::abs(std::real( rhs_thickness)) < T{1.0e-10} );
        assert( column_index < ar.row() );

        typedef T                                       value_type;
        typedef std::size_t                             size_type;
        typedef std::complex<value_type>                complex_type;
        typedef matrix<value_type>                      matrix_type;
        typedef matrix<complex_type>                    complex_matrix_type;

        size_type const n = ar.row();                   //dimension of Ar
        size_type const tilt = diag.row();              //expemental tilts
        size_type const m = ug.size();                  //unknowns in ug

        //create the x, y matrix
        matrix_type x{ n*tilt, (m+m+2) };
        matrix_type y{ intensity };
        y.reshape( n*tilt, 1 );

        //the model intended is 
        // [ c + c0 u0 + c1 u1 + ... + cm um ]^2 + [ d + d0 u0 + d1 u1 + ... + dm um ]^2 = I0
        // ................
        // [ C + C0 u0 + C1 u1 + ... + Cm um ]^2 + [ D + D0 u0 + D1 u1 + ... + Dm um ]^2 = Ix
        //
        // all coefficients goes to matrix x, all right side intensity goes to y, then call  double_square_ug_fitting to fit array u

        //construct matrix A
        complex_matrix_type A{ ar.row(), ar.col() };        // -- move to cache in future
        for_each( A.begin(), A.end(), ar.begin(), [&ug]( complex_type& a, size_type idx ){ a = complex_type{ug[idx][0], 0.0}; } );

        complex_matrix_type rhs_S{ ar.row(), ar.col() };    // -- move to cache in future
        complex_matrix_type column_rhs_S{ ar.row(), 1 };    // -- move to cache in future

        complex_matrix_type lhs_S{ ar.row(), ar.col() };    // -- move to cache in future

        //for every tilt
        for ( size_type tilt_index = 0; tilt_index != tilt; ++ tilt_index )
        {
            std::transform( diag.row_begin(tilt_index), diag.row_end(tilt_index), A.diag_begin(), []( value_type x) { return complex_type{ x, value_type{0} }; } );

            //calculate rhs matrix
            rhs_S = expm( A*rhs_thickness ) ;
            std::copy( rhs_S.col_begin(column_index), rhs_S.col_end( column_index ), column_rhs_S.col_begin(0) );

            //calculate lhs matrix
            coefficient<value_type> coef( lhs_thickness, diag.row_begin(tilt_index), diag.row_end(tilt_index) );
            for ( size_type r = 0; r != n; ++r )
            {
                for ( size_type c = 0; c != n; ++c )
                    lhs_S[r][c] = coef( r, c );
                lhs_S[r][r] = std::exp( rhs_thickness * diag[tilt_index][r] );
            }

            for ( size_type r = 0; r != n; ++r )
            {
                size_type const index_in_x = tilt_index * n + r;
                std::fill( x.row_begin(index_in_x), x.row_end(index_in_x), value_type{0} ); //clear the row

                for ( size_type c = 0; c != n; ++c )
                {
                    complex_type const& ce = lhs_S[r][c] * column_rhs_S[c][0];
                    size_type ar_index = ar[r][c];
                    if ( 0 == ar_index ) // constant
                    {
                        x[index_in_x][0] += std::real( ce );
                        x[index_in_x][m+1] += std::imag( ce );
                    }
                    else
                    {
                        x[index_in_x][ar_index+1] += std::real(ce);
                        x[index_in_x][ar_index+m+2] += std::imag(ce);
                    }
                }

                size_type const index_in_y = index_in_x;
                y[index_in_y][0] = intensity[tilt_index][r];
            }
        }

        double const residual = double_square_ug_fitting( x, y, new_ug );

        std::cout << "residual: " << residual << "\n";

        return residual;
        //check residual
        //if ug and new_ug are close enough, return
        //return
        //otherwise
        //    return make_two_layers_homotopy( ar, diag_begin, intensity, new_ug, lhs_thickness, rhs_thickness, new_ug, intensityi );
    }

    } // namespace homotopy_details


    //IDEA:
    //      after one fitting, merge similiar terms, i.e., modify ar to decrease the unknown variables, and try to fit again  -- only one time?
    //


    template< typename T = double >
    T two_layers_homotopy_forward_fitting(  matrix<std::size_t> const& ar,          // beam index matrix
                                            matrix<T> const& diag,                  // tilt matrix, row-major
                                            matrix<T> const& intensity,             // diffraction measurement, normalized, row-major
                                            matrix<T> const& ug,                    // initial guess
                                            std::complex<T> const& lhs_thickness,   // thickness for left hand side
                                            std::complex<T> const& rhs_thickness,   // thickness for right hand side
                                            matrix<T>& new_ug,                      // fitting result
                                            std::size_t column_index = 0 )          // column index in Scattering matrix
    {
        T current_residual    = homotopy_details::make_two_layers_homotopy( ar, diag, intensity, ug, lhs_thickness, rhs_thickness, new_ug, column_index );
        matrix<T> test_ug{new_ug};
        T new_residual        = homotopy_details::make_two_layers_homotopy( ar, diag, intensity, new_ug, lhs_thickness, rhs_thickness, new_ug, column_index );

        while( new_residual < current_residual )
        {
            current_residual    = new_residual;
            new_residual        = homotopy_details::make_two_layers_homotopy( ar, diag, intensity, new_ug, lhs_thickness, rhs_thickness, new_ug, column_index );
            test_ug             = new_ug;
        }

        new_ug = test_ug;

        return current_residual;
    }

    template< typename T = double >
    T two_layers_homotopy_fitting(  matrix<std::size_t> const& ar, matrix<T> const& diag, matrix<T> const& intensity, matrix<T> const& ug, T const lhs_thickness, T const rhs_thickness, matrix<T>& new_ug, std::size_t column_index = 0 )
    {
        return two_layers_homotopy_fitting( ar, diag, intensity, ug, std::complex<T>{0, lhs_thickness}, std::complex<T>{0, rhs_thickness}, new_ug, column_index );
    }

    template< typename T >
    struct two_layers_homotopy
    {
        typedef T                                           value_type;
        typedef std::complex<value_type>                    complex_type;
        typedef std::size_t                                 size_type;
        typedef matrix<size_type>                           size_matrix_type;
        typedef matrix<value_type>                          matrix_type;
        typedef matrix<complex_type>                        complex_matrix_type;
        typedef c1_data<value_type>                         c1_data_type;
        typedef matrix<c1_data_type>                        c1_matrix_type;
        typedef std::vector<c1_matrix_type>                 c1_matrix_vector_type;
        typedef std::vector<complex_matrix_type>            complex_matrix_vector_type;

        complex_type                                        thickness;

        size_matrix_type                                    ar;
        matrix_type                                         diag;       //store in column order
        matrix_type                                         ug;
        matrix_type                                         initial_ug; //col_major
        matrix_type                                         intensity;  //store in column order

        size_type                                           column_index;

        c1_matrix_vector_type                               c1_matrix_vector;
        complex_matrix_vector_type                          complex_matrix_vector;

        matrix_type                                         x_coef;

        //cache to avoid allocation/deallocation
        complex_matrix_type                                 a_cache;
        complex_matrix_type                                 t_a_cache;
        complex_matrix_type                                 s_cache;
        complex_matrix_type                                 s_column_cache;

        //inner states
        complex_type                                        lhs_thickness;
        complex_type                                        rhs_thickness;

        template< typename Output_Iterator >
        void operator()( Output_Iterator out_it )
        {
            (*this)();
            std::copy( ug.begin(), ug.end(), out_it );
        }

        void operator()()
        {
            check_initial_status();
#if 1 
            size_type const total_steps = 1000;
            size_type const iterations_per_step = 1;

            for ( size_type index = 1; index != total_steps; ++index )
            {
                rhs_thickness = thickness *  static_cast<value_type>(index) / static_cast<value_type>(total_steps);
                lhs_thickness = thickness - rhs_thickness;
                make_iteration( iterations_per_step );

                //std::cout << ug.transpose() << "\n";
                //break;
            }
#endif
#if 0
                rhs_thickness = thickness *  static_cast<value_type>(999) / static_cast<value_type>(1000);
                lhs_thickness = thickness - rhs_thickness;
                make_iteration( 1 );
#endif
        }

        void make_iteration( size_type const iterations )
        {
            make_c1_matrix_vector();

            for ( size_type index = 0; index != iterations; ++index )
            {
                //make_c1_matrix_vector();
                make_complex_matrix_vector();
                make_x_coef_matrix();
                make_ug_refinement();
            }
            //debug code here?
        }

        void make_ug_refinement()
        {
            assert( intensity.row() == x_coef.row() );
            assert( intensity.col() == 1 );
            assert( (x_coef.col() >> 1) - 1  == *std::max_element(ar.begin(), ar.end() ) +1 );

            //insert here?

            double_square_ug_fitting( x_coef, intensity, ug );
        }

        void make_x_coef_matrix()
        {
            size_type const ug_max = *std::max_element( ar.begin(), ar.end() );

            x_coef.clear();

            assert( c1_matrix_vector.size() == complex_matrix_vector.size() );
            assert( c1_matrix_vector.size() == diag.col() );

            for ( size_type index = 0; index != diag.col(); ++index )
                x_coef = x_coef && make_double_square_coef_matrix( c1_matrix_vector[index], complex_matrix_vector[index], ug_max );
                //x_coef = x_coef && make_double_square_coef_matrix( c1_matrix_vector[index], complex_matrix_vector[index], column_index, ug_max );
        }

        //lhs_col_matrhx calculated using c-1 approximation
        void make_c1_matrix_vector()
        {
            c1_matrix_vector.clear();
            for ( size_type index = 0; index != diag.col(); ++index )
                c1_matrix_vector.push_back( make_c1_matrix( lhs_thickness, diag.col_begin(index), diag.col_end(index), ar ) );
        }

        //rhs_col_matrix calculated using standard expm
        void make_complex_matrix_vector()
        {
            if ( ug.size() < initial_ug.size() )
            {
                ug = initial_ug;
            }

            a_cache.resize( ar.row(), ar.col() );
            for ( size_type r = 0; r != ar.row(); ++r )
                for ( size_type c = 0; c != ar.col(); ++c )
                    a_cache[r][c] = complex_type{ ug[ar[r][c]][0], value_type{0} };

            complex_matrix_vector.clear();
            for ( size_type index = 0; index != diag.col(); ++index )
            {
                std::transform( diag.col_begin(index), diag.col_end(index), a_cache.diag_begin(), [](value_type const v){return complex_type{v,0};} );
                //t_a_cache = a_cache*lhs_thickness;
                t_a_cache = a_cache*rhs_thickness;
                s_cache = expm( t_a_cache );
                s_column_cache.resize( s_cache.row(), 1 );
                std::copy( s_cache.col_begin(column_index), s_cache.col_end(column_index), s_column_cache.col_begin(0) );
                complex_matrix_vector.push_back( s_column_cache );
            }
        }

        void check_initial_status()
        {
            assert( std::abs(std::real(thickness)) < value_type{1.0e-10} );
            /*
            assert( std::abs(std::real(lhs_thickness)) < value_type{1.0e-10} );
            assert( std::abs(std::real(rhs_thickness)) < value_type{1.0e-10} );
            */
            assert( std::imag(thickness) > value_type{0} );
            /*
            assert( std::imag(lhs_thickness) > value_type{0} );
            assert( std::imag(rhs_thickness) > value_type{0} );
            */

            assert( ar.row() == ar.col() );
            assert( ar.row() );
            assert( ar.row() == diag.row() );
            assert( *std::max_element(ar.begin(), ar.end()) < initial_ug.size() );
            //assert( ar.row() == intensity.row() );
            assert( ar.row() > column_index );

            //assert( diag.row() == intensity.row() );
            //assert( diag.col() == intensity.col() );
        }

        void config_intensity( matrix_type const& intensity_mat )
        {
            intensity = intensity_mat;
        }

        void config_initial_ug( matrix_type const& ug_mat )
        {
            initial_ug = ug_mat; 
        }

        void config_column_index( size_type const c_index = 0 )
        {
            column_index = c_index;
        }

        void config_diag_matrix( matrix_type const& diag_mat )
        {
            assert( diag_mat.row() );
            assert( diag_mat.col() );

            diag = diag_mat;
        }

        void config_ar_matrix( size_matrix_type const& ar_mat )
        {
            assert( ar_mat.row() == ar_mat.col() );
            assert( ar_mat.row() );
            ar = ar_mat;
        }
/*
        void config_lhs_thickness( complex_type const& c )
        {
            assert( std::abs(std::real(c)) < value_type{1.0e-10} );
            assert( std::imag(c) > value_type{0} );
            lhs_thickness = c;
        }

        void config_rhs_thickness( complex_type const& c )
        {
            assert( std::abs(std::real(c)) < value_type{1.0e-10} );
            assert( std::imag(c) > value_type{0} );
            rhs_thickness = c;
        }

        void config_lhs_thickness( value_type const t )
        {
            assert( t > value_type{0} );
            lhs_thickness = complex_type{0, t };
        }

        void config_rhs_thickness( value_type const t )
        {
            assert( t > value_type{0} );
            rhs_thickness = complex_type{0, t };
        }
*/
        void config_thickness( value_type const t )
        {
            assert( t > value_type{0} );
            thickness = t;
        }

        void config_thickness( complex_type const& c )
        {
            assert( std::abs(std::real(c)) < value_type{1.0e-10} );
            assert( std::imag(c) > value_type{0} );
            thickness = c;
        }
    
    };

}//namespace f

#endif//FYNRKJWVBDGMMXDTKELPDSTMWMLSBJGKRFLTWRFNRYWHURBXEPDMAUUTHIGPLEPNLGOFCKSLY

