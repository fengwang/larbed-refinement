#ifndef OVDJTHNGLVLWGINACNECEOHWQVVNMICLNRKGUJNAQSSNXXUVJUJOFWJDILYNLDSKFSESJPEFE
#define OVDJTHNGLVLWGINACNECEOHWQVVNMICLNRKGUJNAQSSNXXUVJUJOFWJDILYNLDSKFSESJPEFE

#include <f/matrix/matrix.hpp>
#include <f/polynomial/polynomial.hpp>
#include <f/coefficient/coefficient.hpp>

#include <complex>
#include <set>
#include <algorithm>

namespace f
{

    struct simple_c2_polynomial
    {
        typedef unsigned long                                   size_type;
        typedef double                                          value_type;
        typedef std::complex<value_type>                        complex_type;
        typedef simple_symbol<double, size_type>                simple_symbol_type;
        typedef term<value_type, simple_symbol_type>            simple_term_type;
        typedef polynomial<value_type, simple_symbol_type>      simple_polynomial_type;
        typedef matrix<size_type>                               size_matrix_type;
        typedef matrix<value_type>                              matrix_type;
        typedef std::set<size_type>                             zero_set_type;

        //zero set -- some elements shows up there will be ignored in c2 terms
        template<typename Iterator>
        simple_polynomial_type const real( size_matrix_type const& ar, size_type row, size_type col, coefficient<value_type> const& coef, Iterator diag_begin, zero_set_type const& zero_set ) const
        {
            return generate_polynomial( ar, row, col, coef, diag_begin, zero_set, []( complex_type const& c ) { return std::real( c ); });
        }

        template<typename Iterator>
        simple_polynomial_type const imag( size_matrix_type const& ar, size_type row, size_type col, coefficient<value_type> const& coef, Iterator diag_begin, zero_set_type const& zero_set ) const
        {
            return generate_polynomial( ar, row, col, coef, diag_begin, zero_set, []( complex_type const& c ) { return std::imag( c ); });
        }

        template<typename Iterator, typename Coef_Extractor>
        simple_polynomial_type const generate_polynomial( size_matrix_type const& ar, size_type row, size_type col, coefficient<value_type> const& coef, Iterator diag_begin, zero_set_type const& zero_set, Coef_Extractor const& extractor ) const
        {
            simple_polynomial_type answer;
            size_type const n = ar.row();
            assert( ar.row() == ar.col() );
            assert( row < n );
            assert( col < n );

            //C1 part
            size_type const ij_index = ar[row][col];
            size_type const s_ij_index = make_index( ij_index, 0 );
            auto const& rho_ij = make_term<value_type, simple_symbol_type>( simple_symbol_type { value_type{}, s_ij_index } );
            answer += rho_ij * extractor( coef( row, col ) );

            //C2 part
            for ( size_type k = 0; k != n; ++k )
            {
                // ignore if is in zero set
                size_type const ik_index = ar[row][k];
                size_type const kj_index = ar[k][col];
                if ( zero_set.find( ik_index ) != zero_set.end() || zero_set.find( kj_index ) != zero_set.end() )
                {
                    continue;
                }

                value_type c_ikj = extractor( coef( row, k, col ) );

                // it is safe here as we do not use (row==col) intensity
                // i == k != j
                if ( row == k )
                {
                    c_ikj *= *( diag_begin + k );
                    size_type s_kj_index = make_index( kj_index, 0 );
                    simple_term_type const& t_kj = make_term<value_type, simple_symbol_type>( simple_symbol_type { value_type{}, s_kj_index } );
                    answer +=  c_ikj * t_kj;
                    continue;
                }
                // i != k == j
                if ( col == k )
                {
                    c_ikj *= *( diag_begin + k );
                    size_type s_ik_index = make_index( ik_index, 0 );
                    simple_term_type const& t_ik = make_term<value_type, simple_symbol_type>( simple_symbol_type { value_type{}, s_ik_index } );
                    answer +=  c_ikj * t_ik;
                    continue;
                }
                // i != k != j
                size_type const s_ikj_index = make_index( ik_index, kj_index );
                simple_term_type const& t_ikj = make_term<value_type, simple_symbol_type>( simple_symbol_type { value_type{}, s_ikj_index } );
                answer += c_ikj * t_ikj;
            }

            return answer;
        }

        // ( low, high ) -> ( low | high )
        size_type make_index( size_type low = 0, size_type high = 0 ) const
        {
            if ( low > high )
            {
                return make_index( high, low );
            }

            return low << 32 | high;
        }
    };//struct simple_c2_polynomial

}//namespace f

#endif//OVDJTHNGLVLWGINACNECEOHWQVVNMICLNRKGUJNAQSSNXXUVJUJOFWJDILYNLDSKFSESJPEFE

