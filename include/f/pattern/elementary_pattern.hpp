#ifndef DWRSGBUHDRLLQYJFEKFJACJTCUXYYCLHLYAFXNELWWHOFIKJQOJFQMTUKSRBDGCQTNEBSBTTT
#define DWRSGBUHDRLLQYJFEKFJACJTCUXYYCLHLYAFXNELWWHOFIKJQOJFQMTUKSRBDGCQTNEBSBTTT

#include <f/coefficient/coefficient.hpp>
#include <f/coefficient/expm.hpp>
#include <f/matrix/matrix.hpp>
#include <f/nonlinear_optimization/newton_method.hpp>
#include <f/polynomial/polynomial.hpp>
#include <f/polynomial/symbol.hpp>
#include <f/polynomial/complex_symbol.hpp>
#include <f/pattern/index_maker.hpp>
#include <f/variate_generator/variate_generator.hpp>
#include <f/singleton/singleton.hpp>

#include <cassert>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <numeric>
#include <set>
#include <map>
#include <vector>

namespace f
{

struct elementary_pattern
{
    typedef elementary_pattern                              self_type;
    typedef double                                          value_type;
    typedef unsigned long long                              size_type;
    typedef std::complex<value_type>                        complex_type;
    typedef matrix<value_type>                              matrix_type;
    typedef matrix<complex_type>                            complex_matrix_type;
    typedef matrix<size_type>                               size_matrix_type;
    typedef std::string                                     string_type;
    typedef std::set<size_type>                             zero_set_type;
    typedef std::map<size_type, value_type>                 ug_intensity_associate_type;

    typedef complex_symbol<value_type>                      symbol_type;
    typedef term<value_type, symbol_type>                   term_type;
    typedef polynomial<value_type, symbol_type>             polynomial_type;

    typedef simple_symbol<value_type, size_type>            simple_symbol_type;
    typedef term<value_type, simple_symbol_type>            simple_term_type;
    typedef polynomial< value_type, simple_symbol_type>     simple_polynomial_type;

    size_type                                       column_index;               //
    complex_type                                    ipit;                       //thickness * \pi * i
    size_matrix_type                                ar;
    matrix_type                                     diag;                       //column matrix, real, positive

    matrix_type                                     i;                          //the experimental data
    matrix_type                                     i_c1;                       //C1 sim
    matrix_type                                     i_c2;                       //C2 sim
    matrix_type                                     i_c1_c2;                    //C1+C1C2
    matrix_type                                     i_c1_c2_c2;                 //C1+C1C2+C2

    zero_set_type                                   zero_set;
    ug_intensity_associate_type                     ug_intensity;

    polynomial_type                                 c1_polynomial;
    polynomial_type                                 c1_c2_polynomial;
    polynomial_type                                 c2_c2_polynomial;

    polynomial_type                                 power_residual;
    polynomial_type                                 power_residual_without_c1;
    polynomial_type                                 power_residual_without_c1_2;

    simple_polynomial_type const make_simple_real_s_c2_polynomial( size_type const row, size_type const col ) const
    {
        simple_polynomial_type real_part;

        size_type const n = ar.row();
        assert( row < n );
        assert( col < n );

        coefficient<double> const coef( ipit, diag.begin(), diag.end() );

        index_maker& im = singleton<index_maker>::instance();
        matrix_type& ug = singleton<matrix_type>::instance();

        size_type const ij_index = ar[row][col];

        size_type s_index_ij = im.register_key( ij_index );
        //simple_term_type const& rho_ij = make_term< value_type, simple_symbol_type >( simple_symbol_type{ 1.0, s_index_ij } );
        simple_term_type const& rho_ij = make_term< value_type, simple_symbol_type >( simple_symbol_type{ ug[ij_index][0], s_index_ij } );
        complex_type const& c_ij = coef( row, col );
        real_part += rho_ij * std::real( c_ij );

        for ( size_type k = 0; k != n; ++k )
        {
            size_type ik_index = ar[row][k];
            size_type kj_index = ar[k][col];

            if ( zero_set.find( ik_index ) != zero_set.end() )  continue;
            if ( zero_set.find( kj_index ) != zero_set.end() )  continue;

            complex_type c_ikj = coef( row, k, col );

            // i == k != j
            if ( row == k )
            {
                c_ikj *= diag[k][0];
                size_type s_kj_index = im.register_key( kj_index );
                //simple_term_type const& t_kj = make_term<value_type, simple_symbol_type>( simple_symbol_type{ 1.0, s_kj_index } );
                simple_term_type const& t_kj = make_term<value_type, simple_symbol_type>( simple_symbol_type{ ug[kj_index][0], s_kj_index } );
                real_part +=  std::real( c_ikj ) * t_kj;
                continue;
            }
            // i != k == j
            if ( col == k )
            {
                c_ikj *= diag[k][0];
                size_type s_ik_index = im.register_key( ik_index );
                //simple_term_type const& t_ik= make_term<value_type, simple_symbol_type>( simple_symbol_type{ 1.0, s_ik_index } );
                simple_term_type const& t_ik= make_term<value_type, simple_symbol_type>( simple_symbol_type{ ug[ik_index][0], s_ik_index } );
                real_part +=  std::real( c_ikj ) * t_ik;
                continue;
            }

            // i != k != j
            if ( kj_index < ik_index ) std::swap( kj_index, ik_index );
            //size_type const s_ikj_index = im.register_key( (ik_index << 16) + kj_index );
            size_type const s_ikj_index = im.register_key( (ik_index << 16) | kj_index );
            //size_type const s_ikj_index = im.register_key( ik_index,  kj_index );
            //simple_term_type const& t_ikj = make_term<value_type, simple_symbol_type>( simple_symbol_type{ 1.0, s_ikj_index } );
            simple_term_type const& t_ikj = make_term<value_type, simple_symbol_type>( simple_symbol_type{ ug[ik_index][0]*ug[kj_index][0], s_ikj_index } );
            real_part += std::real( c_ikj ) * t_ikj;
        }
        return real_part;
    }

    //TODO -- compare this with C2 sim
    simple_polynomial_type const make_simple_imag_s_c2_polynomial( size_type const row, size_type const col ) const
    {
        simple_polynomial_type imag_part;

        size_type const n = ar.row();
        assert( row < n );
        assert( col < n );

        coefficient<double> const coef( ipit, diag.begin(), diag.end() );

        index_maker& im = singleton<index_maker>::instance();
        matrix_type& ug = singleton<matrix_type>::instance();

        size_type const ij_index = ar[row][col];

        size_type s_index_ij = im.register_key( ij_index );
        //simple_term_type const& rho_ij = make_term< value_type, simple_symbol_type >( simple_symbol_type{ 1.0, s_index_ij } );
        simple_term_type const& rho_ij = make_term< value_type, simple_symbol_type >( simple_symbol_type{ ug[ij_index][0], s_index_ij } );
        complex_type const& c_ij = coef( row, col );
        imag_part += rho_ij * std::imag( c_ij );

        for ( size_type k = 0; k != n; ++k )
        {
            size_type ik_index = ar[row][k];
            size_type kj_index = ar[k][col];

            if ( zero_set.find( ik_index ) != zero_set.end() )  continue;
            if ( zero_set.find( kj_index ) != zero_set.end() )  continue;

            complex_type c_ikj = coef( row, k, col );

            // i == k != j
            if ( row == k )
            {
                c_ikj *= diag[k][0];
                size_type s_kj_index = im.register_key( kj_index );
                //simple_term_type const& t_kj = make_term<value_type, simple_symbol_type>( simple_symbol_type{ 1.0, s_kj_index } );
                simple_term_type const& t_kj = make_term<value_type, simple_symbol_type>( simple_symbol_type{ ug[kj_index][0], s_kj_index } );
                imag_part +=  std::imag( c_ikj ) * t_kj;
                continue;
            }
            // i != k == j
            if ( col == k )
            {
                c_ikj *= diag[k][0];
                size_type s_ik_index = im.register_key( ik_index );
                //simple_term_type const& t_ik= make_term<value_type, simple_symbol_type>( simple_symbol_type{ 1.0, s_ik_index } );
                simple_term_type const& t_ik= make_term<value_type, simple_symbol_type>( simple_symbol_type{ ug[ik_index][0], s_ik_index } );
                imag_part +=  std::imag( c_ikj ) * t_ik;
                continue;
            }

            // i != k != j
            if ( kj_index < ik_index ) std::swap( kj_index, ik_index );
            //size_type const s_ikj_index = im.register_key( (ik_index << 16) + kj_index );
            size_type const s_ikj_index = im.register_key( (ik_index << 16) | kj_index );
            //simple_term_type const& t_ikj = make_term<value_type, simple_symbol_type>( simple_symbol_type{ 1.0, s_ikj_index } );
            simple_term_type const& t_ikj = make_term<value_type, simple_symbol_type>( simple_symbol_type{ ug[ik_index][0]*ug[kj_index][0], s_ikj_index } );
            imag_part += std::imag( c_ikj ) * t_ikj;
        }
        return imag_part;
    }

    //TODO -- compare this with C2 sim
    simple_polynomial_type const make_simple_c2_polynomial( size_type const row, size_type const col ) const
    {

        /*
        simple_polynomial_type real_part;
        simple_polynomial_type imag_part;

        size_type const n = ar.row();
        assert( row < n );
        assert( col < n );

        coefficient<double> const coef( ipit, diag.begin(), diag.end() );
        //matrix_type& ug = singleton<matrix_type>::instance();

        index_maker& im = singleton<index_maker>::instance();

        size_type const ij_index = ar[row][col];

        size_type s_index_ij = im.register_key( ij_index );
        simple_term_type const& rho_ij = make_term< value_type, simple_symbol_type >( simple_symbol_type{ 1.0, s_index_ij } );
        complex_type const& c_ij = coef( row, col );
        real_part += rho_ij * std::real( c_ij );
        imag_part += rho_ij * std::imag( c_ij );

        for ( size_type k = 0; k != n; ++k )
        {
            size_type ik_index = ar[row][k];
            size_type kj_index = ar[k][col];

            complex_type c_ikj = coef( row, k, col );

            // i == k != j
            if ( row == k )
            {
                c_ikj *= diag[k][0];
                size_type s_kj_index = im.register_key( kj_index );
                simple_term_type const& t_kj = make_term<value_type, simple_symbol_type>( simple_symbol_type{ 1.0, s_kj_index } );
                real_part +=  std::real( c_ikj ) * t_kj;
                imag_part +=  std::imag( c_ikj ) * t_kj;
                continue;
            }
            // i != k == j
            if ( col == k )
            {
                c_ikj *= diag[k][0];
                size_type s_ik_index = im.register_key( ik_index );
                simple_term_type const& t_ik= make_term<value_type, simple_symbol_type>( simple_symbol_type{ 1.0, s_ik_index } );
                real_part +=  std::real( c_ikj ) * t_ik;
                imag_part +=  std::imag( c_ikj ) * t_ik;
                continue;
            }

            // i != k != j
            if ( kj_index < ik_index ) std::swap( kj_index, ik_index );
            //size_type const s_ikj_index = im.register_key( (ik_index << 16) + kj_index );
            size_type const s_ikj_index = im.register_key( (ik_index << 16) | kj_index );
            simple_term_type const& t_ikj = make_term<value_type, simple_symbol_type>( simple_symbol_type{ 1.0, s_ikj_index } );
            real_part += std::real( c_ikj ) * t_ikj;
            imag_part += std::imag( c_ikj ) * t_ikj;
        }
        */

        auto const& real_part = make_simple_real_s_c2_polynomial( row, col );
        auto const& imag_part = make_simple_imag_s_c2_polynomial( row, col );

        return real_part * real_part + imag_part * imag_part;
    }

    //TODO -- compare this with C2 sim
    polynomial_type const make_complete_c2_polynomial( size_type const row, size_type const col ) const
    {
        polynomial_type real_part;
        polynomial_type imag_part;

        size_type const n = ar.row();
        assert( row < n );
        assert( col < n );

        coefficient<double> const coef( ipit, diag.begin(), diag.end() );
        matrix_type& ug = singleton<matrix_type>::instance();

        size_type const ij_index = ar[row][col];

        if ( zero_set.find( ij_index ) == zero_set.end() )
        {
            term_type const& rho_ij = make_term< value_type, symbol_type >( make_radius_symbol( ug[ij_index][0], ij_index ) );
            complex_type const& c_ij = coef( row, col );
            real_part += rho_ij * std::real( c_ij );
            imag_part += rho_ij * std::imag( c_ij );
        }

        for ( size_type k = 0; k != n; ++k )
        {
            size_type const ik_index = ar[row][k];
            size_type const kj_index = ar[k][col];

            if ( zero_set.find( ik_index ) != zero_set.end() )  continue;
            if ( zero_set.find( kj_index ) != zero_set.end() )  continue;

            complex_type const& c_ikj = coef( row, k, col );

            term_type const& rho_a_ik = (row == k) ? make_term<value_type, symbol_type>( diag[k][0] ) : make_term<value_type, symbol_type>( make_radius_symbol( ug[ik_index][0], ik_index ) );
            term_type const& rho_a_kj = (col == k) ? make_term<value_type, symbol_type>( diag[k][0] ) : make_term<value_type, symbol_type>( make_radius_symbol( ug[kj_index][0], kj_index ) );

            real_part += rho_a_ik * rho_a_kj * std::real( c_ikj );
            imag_part += rho_a_ik * rho_a_kj * std::imag( c_ikj );
        }

        return real_part * real_part + imag_part * imag_part;
    }

    //TODO check
    //generate a sample polynomial like S_ij
    polynomial_type const make_complete_c2_sample_polynomial( size_type const row, size_type const col ) const
    {
        size_type const n = ar.row();
        assert( row < n );
        assert( col < n );

        matrix_type& ug = singleton<matrix_type>::instance();

        size_type const ij_index = ar[row][col];

        term_type const& rho_ij = make_term< value_type, symbol_type >( make_radius_symbol( ug[ij_index][0], ij_index ) );

        polynomial_type ans = rho_ij;

        for ( size_type k = 0; k != n; ++k )
        {
            size_type const ik_index = ar[row][k];
            size_type const kj_index = ar[k][col];

            term_type const& rho_a_ik = (row == k) ? make_term<value_type, symbol_type>( 1.0 ) : make_term<value_type, symbol_type>( make_radius_symbol( ug[ik_index][0], ik_index ) );
            term_type const& rho_a_kj = (col == k) ? make_term<value_type, symbol_type>( 1.0 ) : make_term<value_type, symbol_type>( make_radius_symbol( ug[kj_index][0], kj_index ) );

            ans += rho_a_ik * rho_a_kj;
        }

        return ans;
    }

    // |C_ikj A_ik A_kj|^2
    polynomial_type const make_c2_c2_polynomial( size_type const row, size_type const col ) const
    {
        polynomial_type ans;

        size_type const n = ar.row();
        assert( row < n );
        assert( col < n );

        matrix_type& ug = singleton<matrix_type>::instance();

        coefficient<double> const coef( ipit, diag.begin(), diag.end() );

        // || C_ikj A_ik A_kj ||^2
        for ( size_type k = 0; k != n; ++k )
        {
            size_type const ik_index = ar[row][k];
            size_type const kj_index = ar[k][col];

            if ( zero_set.find( ik_index ) != zero_set.end() )  continue;
            if ( zero_set.find( kj_index ) != zero_set.end() )  continue;

            complex_type const& c_ikj = coef( row, k, col );

            term_type const& rho_a_ik = (row == k) ? make_term<value_type, symbol_type>( diag[k][0] ) : make_term<value_type, symbol_type>( make_radius_symbol( ug[ik_index][0], ik_index ) );
            term_type const& rho_a_kj = (col == k) ? make_term<value_type, symbol_type>( diag[k][0] ) : make_term<value_type, symbol_type>( make_radius_symbol( ug[kj_index][0], kj_index ) );

            ans += make_term( std::norm(c_ikj), rho_a_ik, rho_a_ik, rho_a_kj, rho_a_kj );
        }
        return ans;
    }

    // 2RE{C_ij A_ij (C_ikj A_ik A_kj)*}
    polynomial_type const make_c1_c2_polynomial( size_type const row, size_type const col ) const
    {
        polynomial_type ans;

        size_type const n = ar.row();
        assert( row < n );
        assert( col < n );

        size_type const ij_index = ar[row][col];

        //ignore if A[r][c] is new zero
        if ( zero_set.find( ij_index ) != zero_set.end() )
            return ans;

        matrix_type& ug = singleton<matrix_type>::instance();

        coefficient<double> const coef( ipit, diag.begin(), diag.end() );
        complex_type const& c_ij  = coef( row, col );
        term_type const& rho_a_ij = make_term<value_type, symbol_type>( make_radius_symbol( ug[ij_index][0], ij_index ) );

        // 2 \sum_k \rho_c_ij \rho_c_ikj \cos(\theta_c_ij-\theta_c_ikj) \rho_ik \rho_kj
        for ( size_type k = 0; k != n; ++k )
        {
            size_type const ik_index = ar[row][k];
            size_type const kj_index = ar[k][col];

            if ( zero_set.find( ik_index ) != zero_set.end() )  continue;
            if ( zero_set.find( kj_index ) != zero_set.end() )  continue;

            complex_type const& c_ikj = coef( row, k, col );

            value_type const rho_rho_cos_2 = 2.0 * std::sqrt( std::norm( c_ij ) * std::norm( c_ikj ) ) * std::cos( std::arg( c_ij ) - std::arg( c_ikj ) );

            term_type const& rho_a_ik = (row == k) ? make_term<value_type, symbol_type>( diag[k][0] ) : make_term<value_type, symbol_type>( make_radius_symbol( ug[ik_index][0], ik_index ) );
            term_type const& rho_a_kj = (col == k) ? make_term<value_type, symbol_type>( diag[k][0] ) : make_term<value_type, symbol_type>( make_radius_symbol( ug[kj_index][0], kj_index ) );

            ans += make_term( rho_rho_cos_2, rho_a_ij, rho_a_ik, rho_a_kj );
        }

        return ans;
    }

    // |C_ij A_ij|^2
    polynomial_type const make_c1_polynomial( size_type const row, size_type const col ) const
    {
        assert( row < ar.row() );
        assert( col < ar.row() );

        size_type const index = ar[row][col];

        if ( zero_set.find( index ) != zero_set.end() )
            return polynomial_type{};

        coefficient<double> const coef( ipit, diag.begin(), diag.end() );

        complex_type const& c_ij = coef( row, col );
        value_type const rho_2 = std::norm( c_ij );

        matrix_type& ug = singleton<matrix_type>::instance();

        term_type const& rho_a_ij = make_term<value_type, symbol_type>( make_radius_symbol( ug[index][0], index ) );
        return rho_a_ij * rho_a_ij * rho_2;
    }

    // [I] - C1^2 - C1C2 -C2^2
    polynomial_type const make_residual() const
    {
        size_type const n = ar.row();

        polynomial_type residual;

        for ( size_type r = 0; r != n; ++r )
        {
            if ( r == column_index ) continue;
            polynomial_type const& c1 = make_c1_polynomial( r, column_index );
            polynomial_type const& c1c2 = make_c1_c2_polynomial( r, column_index );
            polynomial_type const& c2c2 = make_c2_c2_polynomial( r, column_index );
            polynomial_type const diff =  i[r][column_index] - c1 - c1c2 - c2c2;
            residual += diff*diff;
        }
        return residual;
    }

    // [I] - C1^2 - C1C2
    polynomial_type const make_c1_c2_residual() const
    {
        size_type const n = ar.row();

        polynomial_type residual;

        for ( size_type r = 0; r != n; ++r )
        {
            if ( r == column_index ) continue;
            polynomial_type const& c1 = make_c1_polynomial( r, column_index );
            polynomial_type const& c1c2 = make_c1_c2_polynomial( r, column_index );
            polynomial_type const diff =  i[r][column_index] - c1 - c1c2;
            residual += diff*diff;
        }
        return residual;
    }

    // [I-C_1 A] - C1C2
    polynomial_type const make_incomplete_c1_c2_residual() const
    {
        size_type const n = ar.row();

        polynomial_type residual;

        for ( size_type r = 0; r != n; ++r )
        {
            if ( r == column_index ) continue;
            polynomial_type const& c1 = make_c1_polynomial( r, column_index );
            polynomial_type const& c1c2 = make_c1_c2_polynomial( r, column_index );
            polynomial_type const diff =  i[r][column_index] - eval(c1) - c1c2;
            residual += diff*diff;
        }
        return residual;
    }

    // [I - C1 A - C2 ] - C1C2

/*
    // [I - C1^2 - C2^2] - C1C2
    polynomial_type const make_residual_without_c1_c2() const
    {

    }
*/
    void make_zero_set( value_type eps = value_type{1.0e-3} )
    {
        size_type const n = ar.row();

        coefficient<value_type> const coef( ipit, diag.begin(), diag.end() );

        for ( size_type r = 0; r != n; ++r )
        {
            complex_type const& c1 = coef( r, column_index );
            value_type const c1_norm = std::norm( c1 );
            if ( c1_norm < eps*eps*eps*eps ) continue;
            value_type const a_r_c_approx = std::sqrt( i[r][0] / c1_norm );
            if ( a_r_c_approx < eps )
                zero_set.insert( ar[r][column_index] );
        }
    }

    void clear_zero_set()
    {
        zero_set.clear();
    }

    elementary_pattern( size_type column_index_, complex_type const& ipit_, size_matrix_type const& ar_, matrix_type const& diag_ ) : column_index( column_index_ ), ipit( ipit_ ), ar( ar_ ), diag( diag_ )
    {
        assert( column_index < ar.row() );
        assert( ar.row() == ar.col() );
        assert( ar.row() == diag.size() );
        assert( diag.col() == 1 );
    }

    complex_matrix_type const make_a() const
    {
        size_type const n = ar.row();

        complex_matrix_type a{ n, n, complex_type{0.0, 0.0} };

        matrix_type& ug = singleton<matrix_type>::instance();

        for ( size_type r = 0; r != n; ++r )
            for ( size_type c = 0; c != n; ++c )
            {
                if ( r == c ) continue;
                size_type index = ar[r][c];
                assert( index < ug.size() );
                a[r][c] = complex_type{ ug[index][0], value_type{0} };
            }

        for ( size_type index = 0; index != n; ++index )
            a[index][index] = complex_type{ diag[index][0], 0 };

        return a;
    }

    void make_i()
    {
        size_type const n = ar.row();
        i.resize( n, 1 );

        complex_matrix_type const& s = expm( make_a() * ipit );

        std::transform( s.col_begin( column_index ), s.col_end( column_index ), i.begin(), []( complex_type const& c ){ return std::norm(c); } );
    }

    void make_i_c1()
    {
        size_type const n = ar.row();

        i_c1.resize( n, 1 );

        complex_matrix_type const& c1_s = expm( make_a(), ipit, column_index );

        std::transform( c1_s.begin(), c1_s.end(), i_c1.begin(), []( complex_type const& c ){ return std::norm(c); } );
    }

    void make_i_c2()
    {
        size_type const n = ar.row();

        i_c2.resize( n, 1 );

        complex_matrix_type const& c2_s = expm_2( make_a(), ipit, column_index );

        std::transform( c2_s.begin(), c2_s.end(), i_c2.begin(), []( complex_type const& c ){ return std::norm(c); } );
    }

    void make_i_c1_c2()
    {
        size_type const n = ar.row();
        i_c1_c2.resize( n, 1 );

        coefficient<value_type> const coef( ipit, diag.begin(), diag.end() );
        complex_matrix_type const& a = make_a();

        for ( size_type r = 0; r != n; ++r )
        {
            complex_type const& c1 = coef( r, column_index );
            i_c1_c2[r][0] = std::norm( c1 * a[r][column_index] );
            for ( size_type k = 0; k != n; ++k )
            {
                i_c1_c2[r][0] += 2.0 * std::real( c1 * a[r][column_index] * std::conj( coef(r, k, column_index) ) * std::conj(a[r][k]) * std::conj(a[k][column_index]) );
            }
        }
    }

    void make_i_c1_c2_c2()
    {
        size_type const n = ar.row();
        i_c1_c2_c2.resize( n, 1 );

        coefficient<value_type> const coef( ipit, diag.begin(), diag.end() );
        complex_matrix_type const& a = make_a();

        for ( size_type r = 0; r != n; ++r )
        {
            complex_type const& c1 = coef( r, column_index );
            //||A_{i,j} C_{i,j}||^2
            i_c1_c2_c2[r][0] = std::norm( c1 * a[r][column_index] );
            for ( size_type k = 0; k != n; ++k )
            {
                complex_type const& c2 = coef( r, k, column_index );
                // 2 RE{ C1 A_{i,j} C2* A*_{i,k} A*_{k,j} }
                i_c1_c2_c2[r][0] += 2.0 * std::real( c1 * a[r][column_index] * std::conj( c2 ) * std::conj(a[r][k]) * std::conj(a[k][column_index]) );
                // ||C2 A_{i,k} A_{k,j}||^2
                i_c1_c2_c2[r][0] += std::norm( c2 * a[r][k] * a[k][column_index] );
            }
        }
    }

    //ug_intensity
    void make_ug_intensity( value_type eps = 1.0e-50 )
    {
        size_type n = ar.row();
        coefficient<value_type> const coef( ipit, diag.begin(), diag.end() );

        for ( size_type r = 0; r != n; ++r )
        {
            if ( r == column_index ) continue;

            complex_type const& coef_rc = coef( r, column_index );
            value_type const norm2 = std::norm( coef_rc );
            if ( norm2 < eps ) continue;
            value_type const radius = std::sqrt( i[r][column_index] / norm2 );

            size_type const ug_index = ar[r][column_index];
            ug_intensity[ug_index] = radius;
        }
    }

};


}//namespace f

#if 0
struct sto_newton : f::newton_method<double, sto_newton>
{
    typedef std::string                             string_type;
    typedef double                                  value_type;
    typedef std::complex<value_type>                complex_type;
    typedef f::matrix<value_type>                   matrix_type;
    typedef patterns                                pattern_type;
    typedef unsigned long                           size_type;
    typedef f::complex_symbol<value_type>           symbol_type;
    typedef f::polynomial<value_type, symbol_type>  polynomial_type;
    typedef f::matrix<polynomial_type>              p_matrix_type;
    typedef f::matrix<complex_type>                 complex_matrix_type;
    typedef std::vector<symbol_type>                symbol_array_type;
    typedef std::set<size_type>                     potential_zero_set_type;

    size_type                                       ug_length;
    pattern_type                                    pattern;
    value_type                                      total_power;

    matrix_type                                     polar_ug;
    p_matrix_type                                   hessian;
    symbol_array_type                               symbol_array;
    p_matrix_type                                   jacobi;

    matrix_type                                     test_i;

    polynomial_type                                 merit_function;

    polynomial_type                                 power_residual;
    polynomial_type                                 radius_residual;

    potential_zero_set_type                         potential_zero_set;

    sto_newton( size_type ug_length_ ) : ug_length( ug_length_ )
    {
        polar_ug.resize( ug_length_, 1 );
        make_symbol_array();
    }

    void load_pattern( pattern_type const& pattern_ )
    {
        pattern = pattern_;
        make_accurate_polar_ug();
    }

    void make_radius_residual( value_type weigh = 1.0e-5 )
    {
        radius_residual.clear();
        polynomial_type ans;
        for ( auto const& element : pattern.ug_intensity )
        {
            size_type ij_index = element.first;
            value_type value = element.second;
            auto const& rho_a_ij = f::make_term<value_type>( f::make_radius_symbol( polar_ug[ij_index][0], ij_index ) );
            ans = rho_a_ij * rho_a_ij - value * value;
            radius_residual +=  weigh * ans * ans;
        }
    }

    void make_power_residual()
    {
        power_residual.clear();

        for ( unsigned long r = 0; r != pattern.i.row(); ++r )
            for ( unsigned long c = 0; c != pattern.i.col(); ++c )
            {
                if ( r == pattern.column_index ) continue;
                auto const& prc_diff =  make_intensity_polynomial( r, c ) - pattern.i[r][c];
                power_residual +=  prc_diff * prc_diff;
            }

        power_residual += f::make_term<value_type, symbol_type>( value_type {1.0e-10} );

        std::cout << "\nthe power_residual is \n" << power_residual << "\n";
        std::cout << "\nthe power_residual is evaluated as \n" << f::eval( power_residual) << "\n";
    }



    void on_fitting_start()
    {
        std::cerr << "\nAt the global solution, the power residual is " << f::eval( power_residual ) << "\n";
        ( ( *this ).array_x ).resize( ug_length, 1 );
    }

    void on_iteration_start( unsigned long n  )
    {
        std::copy( ( ( *this ).array_x ).begin(), ( ( *this ).array_x ).end(), polar_ug.begin() );

        //fill zero elemented generated by C1 fitting
        for ( auto itor = pattern.zero_set.begin(); itor != pattern.zero_set.end(); ++itor )
        {
            size_type index = *itor;
            polar_ug[index][0] = value_type {};
        }

        //fill potential zero set
        for ( auto const& index : potential_zero_set )
            polar_ug[index][0] = value_type{};

        //move to zero if too large
        std::for_each( polar_ug.begin(), polar_ug.end(), [this]( value_type& x ) 
                                                         { 
                                                            while ( std::abs(x) > 0.1 )
                                                            {
                                                                if ( x > 0.1 ) x = 0.2 - x;
                                                                if ( x < -0.1 ) x = -0.2 - x; 
                                                            }
                                                         } 
                     );

        std::copy( polar_ug.begin(), polar_ug.end(), ( ( *this ).array_x ).begin() );

        std::cout << "\nat step " << n << "\n";
        std::cout << "\nthe fitting result is \n" << polar_ug << "\n";
        std::cout << "\nthe merit_function residual is \t" << f::eval( merit_function ) << "\n";
        std::cout << "\nthe power residual is \t" << f::eval( power_residual ) << "\n";
        std::cout << "\nthe radius residual is \t" << f::eval( radius_residual ) << "\n";
    }

    void on_iteration_end( unsigned long )
    {
    }

    bool stop_here()
    {
        if ( f::eval( power_residual ) < 1.0e-7 ) 
        {
            std::cout << "\nstop as accuracy reached.\n";
            return true;
        }

        if ( std::any_of( polar_ug.begin(), polar_ug.end(), []( value_type x ) { return std::isinf( x ) || std::isnan( x ); } ) )
        {
            std::cout << "\nstop as nan/inf found.\n";
            return true;
        }

        return false;
    }

    void on_fitting_end()
    {
        std::copy( ( ( *this ).array_x ).begin(), ( ( *this ).array_x ).end(), polar_ug.begin() );
    }

    //initialization before fitting
    void generate_initial_guess( value_type* x ) const
    {
        f::variate_generator<value_type> vg( -0.1, 0.1 );
        for ( size_type index = 0; index != ug_length; ++index )
        {
            *( x + index ) = vg();
        }
    }

    unsigned long unknown_variable_size() const
    {
        return ug_length;
    }

    value_type gamma() const
    {
        return 0.01618;
    }

    unsigned long loops()
    {
        return 1000;
    }

    value_type calculate_derivative( const unsigned long n, value_type* ) const
    {
        return f::eval( jacobi[n][0] );
    }

    value_type calculate_second_derivative( const unsigned long r, const unsigned long c, value_type* ) const
    {
        return f::eval( hessian[r][c] );
    }

    void make_hessian()
    {
        hessian.resize( ug_length, ug_length );
        for ( size_type r = 0; r != jacobi.row(); ++r )
            for ( size_type c = 0; c != jacobi.row(); ++c )
            {
                if ( r > c ) continue;
                hessian[r][c] = f::make_polynomial_derivative( jacobi[r][0], symbol_array[c] );
                hessian[c][r] = hessian[r][c];
            }
        std::cerr << "\nhessian matrix generated.\n";
    }

    void make_jacobi()
    {
        jacobi.resize( ug_length, 1 );
        potential_zero_set.clear();

        for ( size_type index = 0; index != ug_length; ++index )
        {
            jacobi[index][0] = f::make_polynomial_derivative( merit_function, symbol_array[index] );

            if ( 0 == jacobi[index][0].size() ) potential_zero_set.insert( index );
            std::cout << "\nthe jacobi matrix " << index << "(" << jacobi[index][0].size() << ") is:\t" << jacobi[index][0] << "\n";
        }
        std::cerr << "\njacobi matrix generated.\n";
        std::cerr << "\nFound " << potential_zero_set.size() << " zero elements.\n";
        //std::cerr << "\nthe potential zero set are\n";
        //for ( auto const& element : potential_zero_set )
        //    std::cerr << element << "\n";
    }

    void make_symbol_array()
    {
        symbol_array.reserve( ug_length );
        for ( size_type index = 0; index != ug_length; ++index )
            symbol_array.push_back( f::make_radius_symbol( polar_ug[index][0], index ) );
    }

    void make_accurate_polar_ug()
    {
        for ( size_type r = 0; r != ug_length; ++r )
            polar_ug[r][0] = std::real( pattern.ug[r][0] );
    }

    void make_test_i()
    {
        test_i.resize( pattern.i.row(), pattern.i.col() );
        for ( size_type r = 0; r != test_i.row(); ++r )
            for ( size_type c = 0; c != test_i.col(); ++c )
                test_i[r][c] = f::eval( make_intensity_polynomial( r, c ) );
    }

#if 1
    //generate the polynomial expression coresponding to I( row, col )
    polynomial_type const make_intensity_polynomial( size_type const row, size_type const col ) const
    {
        assert( row < pattern.i.row() );
        assert( col < pattern.i.col() );

        polynomial_type ans;

        size_type const row_index = row;
        size_type const col_index = pattern.column_index;

        f::coefficient<double> const coef( pattern.ipit, pattern.offset_2.col_begin( col ), pattern.offset_2.col_end( col ) );

        complex_type const& c_ij = coef( row_index, col_index );
        value_type rho_2 = std::norm( c_ij );

        size_type const ij_index = ( pattern.ar )[row_index][col_index];

        auto const& rho_a_ij = f::make_term<value_type>( f::make_radius_symbol( polar_ug[ij_index][0], ij_index ) );
        //auto const& cos_a_ij = f::make_term<value_type>( f::make_cosine_symbol( polar_ug[ij_index][1], ij_index ) );

        if ( pattern.zero_set.find( pattern.ar[row_index][col_index] ) == pattern.zero_set.end() )
        {

            ans += f::make_term( rho_2, rho_a_ij, rho_a_ij );

            for ( size_type k = 0; k != pattern.a.row(); ++k )
            {

                complex_type const& c_ikj = coef( row_index, k, col_index );

                value_type const rho_rho_cos_2 = 2.0 * std::sqrt( std::norm( c_ij ) * std::norm( c_ikj ) ) * std::cos( std::arg( c_ij ) - std::arg( c_ikj ) );

                size_type const ik_index = pattern.ar[row_index][k];
                size_type const kj_index = pattern.ar[k][col_index];

                if ( pattern.zero_set.find( ik_index ) != pattern.zero_set.end() )  continue;
                if ( pattern.zero_set.find( kj_index ) != pattern.zero_set.end() )  continue;

                //binding here
                auto rho_a_ik = f::make_term<value_type>( f::make_radius_symbol( polar_ug[ik_index][0], ik_index ) );
                //auto cos_a_ik = f::make_term<value_type>( f::make_cosine_symbol( polar_ug[ik_index][1], ik_index ) );

                if ( row_index == k )
                {
                    rho_a_ik = f::make_term<value_type, symbol_type>( std::real( pattern.offset_2[row_index][col] ) );
                    //cos_a_ik = f::make_term<value_type, symbol_type>( 1.0 );
                }

                auto rho_a_kj = f::make_term<value_type>( f::make_radius_symbol( polar_ug[kj_index][0], kj_index ) );
                //auto cos_a_kj = f::make_term<value_type>( f::make_cosine_symbol( polar_ug[kj_index][1], kj_index ) );

                if ( k == col_index )
                {
                    rho_a_kj = f::make_term<value_type, symbol_type>( std::real( pattern.offset_2[col_index][col] ) );
                    //cos_a_kj = f::make_term<value_type, symbol_type>( 1.0 );
                }

                //ans += f::make_term( rho_rho_cos_2, rho_a_ij, rho_a_ik, rho_a_kj, cos_a_ij, cos_a_ik, cos_a_kj );
                ans += f::make_term( rho_rho_cos_2, rho_a_ij, rho_a_ik, rho_a_kj );

            }
        }
        return ans;
    }
#endif

};//sto_newton

#endif


#if 0

int main()
{
    typedef double                                  value_type;
    typedef f::matrix<value_type>                   matrix_type;
    typedef std::complex<value_type>                complex_type;
    typedef f::matrix<complex_type>                 complex_matrix_type;
    typedef f::complex_symbol<value_type>           symbol_type;
    typedef f::polynomial<value_type, symbol_type>  polynomial_type;
    typedef unsigned long                           size_type;
    typedef std::set<size_type>                     zero_set_type;

    //load ug
    matrix_type raw_ug;
    raw_ug.load( "data/STO/ug.txt" );

    size_type n = raw_ug.row();

    complex_matrix_type the_ug( raw_ug.row(), 1 );
    for ( unsigned long r = 0; r != raw_ug.row(); ++r )
        the_ug[r][0] = std::complex<double>{ raw_ug[r][1], raw_ug[r][2] };

    std::cout << "\nthe generated ug is \n" << the_ug << "\n";

    std::cout.precision( 15 );
    std::complex<double> ipit { 0.0, 1.0 };

    unsigned long total_pattern = 29;
    //unsigned long total_pattern = 1;
    unsigned long column_index = 0;

    polynomial_type total_power_residual;
    polynomial_type total_radius_residual;
    zero_set_type total_zero_set;
//also need to store zero element

    patterns pt;
    sto_newton sn( n );

    //collect all the zero sets
    for ( unsigned long index = 0; index != total_pattern; ++index )
    {
        std::string const& id = boost::lexical_cast<std::string>( index );
        std::string const& ar_dat = "data/STO/Ar_" + id + ".txt"; 
        std::string const& offset_dat = "data/STO/Diag_" + id + ".txt"; 

        pt.load_ar( ar_dat );
        pt.load_offset( offset_dat );
        pt.load_ipit( ipit );
        pt.load_ug( the_ug );
        pt.load_column( column_index );
        pt.refine_with_sim2();

        total_zero_set.insert( pt.zero_set.begin(), pt.zero_set.end() );

        sn.load_pattern( pt );
        sn.make_power_residual();
        sn.make_radius_residual();
        total_power_residual += sn.power_residual;
        total_radius_residual += sn.radius_residual;
    }

    //sn.merit_function = total_power_residual;
    sn.merit_function = total_power_residual + total_radius_residual;
    sn.power_residual = total_power_residual;
    sn.pattern.zero_set = total_zero_set;

    // add C1 elements here


    sn.make_jacobi();
    sn.make_hessian();

    //fitting
    std::vector<double> ans;
    ans.resize( n );
    sn( ans.begin() );
    std::cout.precision( 15 );

    sn.make_accurate_polar_ug();

    std::cout << "\nCompare:\nfitted \t\t original \t\t difference\n";
    for ( size_type i = 0; i != n; ++i )
        std::cout << ans[i] << "\t\t" << sn.polar_ug[i][0] << "\t\t" << ans[i] - sn.polar_ug[i][0] << "\n";

    return 0;
}
#endif // 


#endif//DWRSGBUHDRLLQYJFEKFJACJTCUXYYCLHLYAFXNELWWHOFIKJQOJFQMTUKSRBDGCQTNEBSBTTT

