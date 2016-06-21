#ifndef XUUWPTKJTQAHXVHIPGBIFGARYBBCGXDOYWSUWRJBTOWOFYEHNPQTUTTIVWETWRASWKKIQHEHR
#define XUUWPTKJTQAHXVHIPGBIFGARYBBCGXDOYWSUWRJBTOWOFYEHNPQTUTTIVWETWRASWKKIQHEHR

#include <f/pattern/pattern.hpp>
#include <f/coefficient/coefficient.hpp>
#include <f/algorithm/for_each.hpp>

#include <f/dynamic_inverse/impl/scattering_matrix.hpp>

#include <vector>
#include <cmath>
#include <cassert>

namespace f
{

    template< typename T >
    struct multi_c1
    {
        typedef T               value_type;

        pattern<T> const&       pattern_;
        unsigned long           order_;

        std::vector<matrix<std::complex<value_type> > > coef_c1;        //coeficient for each tilt

        std::vector<matrix<std::complex<value_type> > > s_cache;        //layer-1 cache for each tilt

        std::vector<matrix<std::complex<value_type> > > col_s_cache;    //layer-2 cache for each tilt

        std::vector<matrix<std::complex<value_type> > > S_cache;        //scattering matrix for each tilt

        value_type square_residual( value_type* x__ ) 
        {
            double ac = x__[0];
            double dc = x__[1];

            for ( unsigned long index = 0; index != pattern_.tilt_size; ++index )
                update_scattering_matrix( index, x__ );

            value_type residual = value_type{0};
            for ( unsigned long index = 0; index != pattern_.tilt_size; ++index )
                residual += make_square_residual( index, ac, dc );

            return residual;
        }

        value_type abs_residual( value_type* x__ )
        {
            double ac = x__[0];
            double dc = x__[1];

            for ( unsigned long index = 0; index != pattern_.tilt_size; ++index )
                update_scattering_matrix( index, x__ );

            value_type residual = value_type{0};
            for ( unsigned long index = 0; index != pattern_.tilt_size; ++index )
                residual += make_abs_residual( index, ac, dc );

            return residual;
        }

        auto make_merit_function()
        {
            return [this]( value_type* x__ ){ return (*this).square_residual( x__ ); };
        }

        auto make_abs_function()
        {
            return [this]( value_type* x__ ){ return (*this).abs_residual( x__ ); };
        }

        multi_c1( pattern<T> const& pattern__, unsigned long order__ ) : pattern_(pattern__), order_( order__ ) 
        {
            assert( order__ );
            auto const& thickness = std::complex<value_type>{ std::real(pattern_.thickness)/static_cast<value_type>(order_), std::imag(pattern_.thickness)/static_cast<value_type>(order_) };

            for ( unsigned long index = 0; index != pattern_.tilt_size; ++index )
            {
                unsigned long const dim = ((pattern_.ar)[index]).row();
                coefficient<value_type> const coef{ thickness, ((pattern_.diag)[index]).begin(), ((pattern_.diag)[index]).end() };
                matrix<std::complex<value_type> > coef_mat( dim, dim );
                for ( unsigned long row = 0; row != dim; ++row )
                {
                    for ( unsigned long col = 0; col != dim; ++col )
                        coef_mat[row][col] = coef( row, col );

                    coef_mat[row][row] = std::exp( thickness * (pattern_.diag)[index][row][0] );
                }

                coef_c1.emplace_back( coef_mat );

                s_cache.emplace_back( dim, dim ) ;
                col_s_cache.emplace_back( dim, 1 );
                S_cache.emplace_back( dim, 1 );
            }
        }

        //private area

        void update_s( unsigned long index, value_type* x__ ) 
        {
            assert( index < pattern_.tilt_size );
            assert( s_cache.size() == pattern_.tilt_size );
            assert( col_s_cache.size() == pattern_.tilt_size );
            assert( S_cache.size() == pattern_.tilt_size );
            assert( s_cache[index].row() == s_cache[index].col() );
            assert( s_cache[index].row() == (pattern_.ar)[index].row() );
#if 1 
            //update s
            unsigned long const dim = (pattern_.ar)[index].row();
            for ( unsigned long r = 0; r != dim; ++r )
            {
                double res = 0.0;
                for ( unsigned long c = 0; c != dim; ++c )
                {
                    if ( r == c ) continue;

                    unsigned long const ar_index = (pattern_.ar)[index][r][c];
                    unsigned long const real_offset = ar_index + ar_index;
                    unsigned long const imag_offset = real_offset + 1;

                    s_cache[index][r][c] = std::complex<value_type>{ x__[real_offset], x__[imag_offset] } * coef_c1[index][r][c];
                    res += std::norm( s_cache[index][r][c] );
                }
                if ( res >= 1.0 ) 
                    s_cache[index][r][r] = std::complex<value_type>{0.0, 0.0};
                else
                    s_cache[index][r][r] = coef_c1[index][r][r] * std::sqrt( 1.0 - res );
            }
#endif
#if 0 
            matrix<std::complex<value_type>> ug( pattern_.ug_size, 1 );
            std::copy( x__, x__+pattern_.ug_size*2, reinterpret_cast<value_type*>( ug.data() ) );
            s_cache[index] = make_scattering_matrix_c1( (pattern_.ar)[index], ug, pattern_.diag[index].begin(), pattern_.diag[index].end(), pattern_.thickness / static_cast<value_type>(order_) );
#endif
        }

        void update_col_s( unsigned long index )
        {
            assert( col_s_cache[index].row() == s_cache[index].row() );
            assert( col_s_cache[index].col() == 1 );

            std::copy( s_cache[index].col_begin(0), s_cache[index].col_end(0), col_s_cache[index].begin() );
        }

        void update_S( unsigned long index ) // S = s * _s;
        {
            assert( s_cache[index].row() == col_s_cache[index].row() );
            assert( 1 == col_s_cache[index].col() );
            assert( s_cache[index].row() == S_cache[index].row() );
            assert( 1 == S_cache[index].col() );

            unsigned long const dim = s_cache[index].row();
            for ( unsigned long idx = 0; idx != dim; ++idx )
                S_cache[index][idx][0] = std::inner_product( s_cache[index].row_begin(idx), s_cache[index].row_end(idx), col_s_cache[index].begin(), std::complex<value_type>{0, 0} );
        }

        void swap( unsigned long index )
        {
            S_cache[index].swap( col_s_cache[index] );
        }

        //all to S_cache
        void update_scattering_matrix( unsigned long index, value_type* x__ )
        {
            update_s( index, x__ );;
            update_col_s( index );
            for ( unsigned long jndex = 1; jndex != order_; ++jndex )
            {
                update_S( index );
                swap( index );
            }
            swap( index );
        }

        value_type make_square_residual( unsigned long index )
        {
            auto const & s = S_cache[index];
            auto const & I = (pattern_.intensity)[index];

            //value_type total_power = value_type{0};
            //for_each( s.begin(), s.end(), [&total_power]( auto const& s_ ) { total_power += std::norm(s_); } );

            value_type residual = value_type{0};
            //for_each( s.begin(), s.end(), I.begin(), [&residual,total_power]( auto const& s_, auto const& i_ ){ auto diff = std::norm(s_)/total_power - i_; residual += diff * diff; } );
            for_each( s.begin(), s.end(), I.begin(), [&residual]( auto const& s_, auto const& i_ ){ auto diff = std::norm(s_) - i_; residual += diff * diff; } );

            return residual;
        }

        value_type make_square_residual( unsigned long index, double ac, double dc )
        {
            auto const & s = S_cache[index];
            auto const & I = (pattern_.intensity)[index];

            value_type residual = value_type{0};
            //for_each( s.begin(), s.end(), I.begin(), [&residual,total_power, ac, dc]( auto const& s_, auto const& i_ ){ auto diff = dc + ac * std::norm(s_)/total_power - i_; residual += diff * diff; } );
            for_each( s.begin(), s.end(), I.begin(), 
                      [&residual, ac, dc]( auto const& s_, auto const& i_ )
                      { 
                        auto const diff = std::norm(s_) - i_; 
                        //auto const diff = dc + ac * std::norm(s_) - i_; 
                        auto const res = diff * diff  / ( 1.0 + std::exp( 12.56637061435917295384 * i_ ) ) ;
                        residual += res;
                      } 
                    );

            return residual;
        }

        value_type make_abs_residual( unsigned long index )
        {
            auto const & s = S_cache[index];
            auto const & I = (pattern_.intensity)[index];

            value_type total_power = value_type{0};
            for_each( s.begin(), s.end(), [&total_power]( auto const& s_ ) { total_power += std::norm(s_); } );

            value_type residual = value_type{0};
            for_each( s.begin(), s.end(), I.begin(), [&residual,total_power]( auto const& s_, auto const& i_ ){ auto diff = std::norm(s_)/total_power - i_; residual += std::abs(diff); } );
            return residual;
        }

        value_type make_abs_residual( unsigned long index, double ac, double dc )
        {
            auto const & s = S_cache[index];
            auto const & I = (pattern_.intensity)[index];

            value_type total_power = value_type{0};
            for_each( s.begin(), s.end(), [&total_power]( auto const& s_ ) { total_power += std::norm(s_); } );

            value_type residual = value_type{0};
            for_each( s.begin(), s.end(), I.begin(), [&residual,total_power, ac, dc]( auto const& s_, auto const& i_ ){ auto diff = dc + ac*std::norm(s_)/total_power - i_; residual += std::abs(diff); } );
            return residual;
        }


    };

    template< typename T >
    auto make_multi_c1( pattern<T> const& pattern__, unsigned long order__ )
    {
        return multi_c1<T>{ pattern__, order__ };
    }

}//namespace f

#endif//XUUWPTKJTQAHXVHIPGBIFGARYBBCGXDOYWSUWRJBTOWOFYEHNPQTUTTIVWETWRASWKKIQHEHR

