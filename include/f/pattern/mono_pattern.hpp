#ifndef VVJLQHRRJAOGOCBWVCAVSIWCPFUWUMSOXQDADYGRKDCFHOYKXQUCTDFHXRHQDRLROLEITMWLD
#define VVJLQHRRJAOGOCBWVCAVSIWCPFUWUMSOXQDADYGRKDCFHOYKXQUCTDFHXRHQDRLROLEITMWLD

#include <f/algorithm/for_each.hpp>
#include <f/numeric/inner_product.hpp>
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

struct mono_pattern
{
    typedef mono_pattern                                    self_type;
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

    size_type                                               pattern_size;
    size_type                                               column_index;
    complex_type                                            ipit;                       //thickness * \pi * i
    size_matrix_type                                        ar;                         //all patterns share a same Ar
    matrix_type                                             diag;                       //column matrix, real, positive, [n][pattern_size]
    matrix_type                                             i;                          //the experimental data, [n][pattern_size]

    ug_intensity_associate_type                             c1_approximation;           //associated of [index][value]

    void make_c1_approximation()
    {
        //the approximation is I = ||C(r,c) A(r,c)||^2
        //build a matrix of coefficient coef(r,c)
        matrix_type abs_co{diag.row(), diag.col()};
        for ( size_type c = 0; c != abs_co.col(); ++c )
        {
            coefficient<double> const coef{ ipit, diag.col_begin(c), diag.col_end(c) };
            for ( size_type r = 0; r != abs_co.row(); ++r )
                abs_co[r][c] = std::abs( coef(r, column_index) );
        }

        //[total_{weigh*approximation}] -- [total_weigh] ---->>>> here we use intensity as weigh
        matrix_type weighted_approximation{ar.row(), 2, double{0}};
        for ( size_type r = 0; r != weighted_approximation.row(); ++r )
        {
            if ( r == column_index ) continue;
            weighted_approximation[r][0] = inner_product( i.row_begin(r), i.row_end(r), abs_co.row_begin(r), double{0}, [](double intensity, double coefficient){ return intensity*std::sqrt(intensity/coefficient); } );
            weighted_approximation[r][1] = std::accumulate( i.row_begin(r), i.row_end(r), double{0} );
        }

        //generate the c1 approximation result
        size_type min_index = *std::min_element( ar.col_begin(column_index), ar.col_end(column_index) );
        size_type max_index = *std::max_element( ar.col_begin(column_index), ar.col_end(column_index) );

        //low efficient, use map in later case
        for ( size_type index = min_index; index != max_index+1; ++index )
        {
            if ( index == column_index ) continue;

            double wapp = 0.0;
            double weig = 0.0;
            for ( size_type r = 0; r != weighted_approximation.row(); ++r )
            {
                if ( r != index ) continue;
                wapp += weighted_approximation[r][0];
                weig += weighted_approximation[r][1];
            }
            c1_approximation[index] = wapp/weig;
        }
    }

    mono_pattern( size_type pattern_size_, size_type column_index_, complex_type const& ipit_, string_type const& folder_path_ ) : pattern_size( pattern_size_ ), column_index( column_index_ ), ipit( ipit_ )
    {
        //load ar
        string_type const& ar_path = folder_path_ + string_type{"Ar.txt"};
        ar.load( ar_path );

        //load diag
        size_type n = ar.row();
        assert( ar.row() == ar.col() );
        diag.resize( n, pattern_size );
        matrix_type diag_tmp{ n, 1 };
        for ( size_type c = 0; c != pattern_size; ++c )
        {
            string_type const& diag_path = folder_path_ + string_type{"Diag_"} + std::to_string(c) + string_type{".txt"};
            diag_tmp.load( diag_path );
            std::copy( diag_tmp.col_begin(0), diag_tmp.col_end(0), diag.col_begin(c) );
        }

        //load intensity
    }

    void make_simulated_intensity()
    {
        size_type const n = ar.row();
        i.resize( n, pattern_size );

        auto a = make_a();

        for ( size_type index = 0; index != pattern_size; ++index )
        {
            std::copy( diag.col_begin(index), diag.col_end(index), a.diag_begin() );
            auto const& s = expm( a*ipit );
            std::transform( s.col_begin( column_index ), s.col_end( column_index ), i.col_begin(index), []( complex_type const& c ){ return std::norm(c); } );
        }
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
        simple_term_type const& rho_ij = make_term< value_type, simple_symbol_type >( simple_symbol_type{ ug[ij_index][0], s_index_ij } );
        complex_type const& c_ij = coef( row, col );
        real_part += rho_ij * std::real( c_ij );

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
        auto const& real_part = make_simple_real_s_c2_polynomial( row, col );
        auto const& imag_part = make_simple_imag_s_c2_polynomial( row, col );

        return real_part * real_part + imag_part * imag_part;
    }

};//struct mono_pattern

}//namespace f

#endif//VVJLQHRRJAOGOCBWVCAVSIWCPFUWUMSOXQDADYGRKDCFHOYKXQUCTDFHXRHQDRLROLEITMWLD

