#ifndef XBWEAARGAAOAJHJCYFVFUKARCELUWRCBXFFRPGYVAIQHPNMELXAANWWVCQXIKKGDYFJCQUPTS
#define XBWEAARGAAOAJHJCYFVFUKARCELUWRCBXFFRPGYVAIQHPNMELXAANWWVCQXIKKGDYFJCQUPTS

#include <f/algorithm/min.hpp>
#include <f/algorithm/max.hpp>

#include <cmath>
#include <cassert>
#include <cstddef>
#include <vector>
#include <iterator>
#include <algorithm>
#include <complex>

namespace f
{

    template< typename T >
    struct coefficient
    {
        typedef T                               value_type;
        typedef std::complex<value_type>        result_type;
        typedef std::size_t                     size_type;
        typedef std::vector<result_type>        array_type;

        result_type                             thickness;
        array_type                              diagonal;
        value_type                              eps;

        template< typename Input_Iterator >
        coefficient( value_type thickness_, Input_Iterator first_, Input_Iterator last_, const value_type eps_ = value_type(1.0e-9) ) : thickness( result_type{0, thickness_} ), diagonal( first_, last_ ), eps( eps_ )
        {
        }

        template< typename Input_Iterator >
        coefficient( result_type thickness_, Input_Iterator first_, Input_Iterator last_, const value_type eps_ = value_type(1.0e-9) ) : thickness( thickness_ ), diagonal( first_, last_ ), eps( eps_ )
        {
        }

        size_type size() const
        {
            return diagonal.size();
        }

        result_type delta( const size_type m, const size_type n ) const
        {
            assert( m < size() );
            assert( n < size() );
            return m == n ? result_type(1.0, 0.0) : result_type(0.0, 0.0);
        }

        result_type operator()( const size_type n, const size_type m ) const
        {
            assert( n < size() );
            assert( m < size() );

            if ( n == m )
                return result_type(0.0, 0.0);

            const result_type bn = diagonal[n];
            const result_type bm = diagonal[m];

            if ( std::abs(bn - bm) > eps )
                return ( std::exp( thickness*bn ) - std::exp( thickness*bm ) ) / ( bn - bm );

            return thickness * std::exp( thickness * bn );
        }

        result_type operator()( const size_type n, const size_type l, const size_type m ) const
        {
            assert( n < size() );
            assert( l < size() );
            assert( m < size() );

            auto const& complex_compare = []( const result_type& lhs, const result_type& rhs ) { return std::abs(lhs) < std::abs(rhs); };
            result_type const& z = max( diagonal[n], diagonal[m], diagonal[l], complex_compare );
            result_type const& x = min( diagonal[n], diagonal[m], diagonal[l], complex_compare );

            result_type const& y = diagonal[n] + diagonal[l] + diagonal[m] - x - z;

            if ( std::abs( x - y ) < eps )
            {
                if ( std::abs( y - z ) < eps )
                    return c2_1( x ); // x = y = z

                return c2_2( x, z ); // x = y != z
            }

            if ( std::abs( y - z ) < eps )
                return c2_2( z, x ); //x != y = z

            return c2_3( x, y, z ); //x != y != z
        }

    private:
        result_type c2_1( const result_type& x ) const
        {
            return thickness * thickness * std::exp( x * thickness ) / value_type( 2.0 );
        }

        result_type c2_2( const result_type& x, const result_type& y ) const
        {
            const result_type& one   = value_type( 1.0 );
            const result_type& tx    = thickness * x;
            const result_type& ty    = thickness * y;
            const result_type& ex    = std::exp(tx);
            const result_type& ey    = std::exp(ty);
            const result_type& xy    = x - y;
            const result_type& xxyy  = xy * xy;

            const result_type& ansx  = ( tx - one ) * ex;
            const result_type& ansy  = ( ty - one ) * ey;

            return ( ansx - ansy ) / xxyy;

            /*
            const result_type& ex    = std::exp( tx ) - one;
            const result_type& ey    = std::exp( ty ) - one;
            const result_type& xy    = x - y;
            const result_type& xxyy  = xy * xy;
            const result_type& ans_x = - ex + tx;
            const result_type& ans_y = ( tx - ty ) * ex;
            const result_type& ans_z = ey - ty;

            return ( ans_x + ans_y + ans_z ) / xxyy;
            */
        }

        result_type c2_3( const result_type& x, const result_type& y, const result_type& z ) const
        {
            const result_type& one   = value_type( 1.0 );
            const result_type& tx    = thickness * x;
            const result_type& ty    = thickness * y;
            const result_type& tz    = thickness * z;
            const result_type& ex_   = std::exp( tx ) - one;
            const result_type& ey_   = std::exp( ty ) - one;
            const result_type& ez_   = std::exp( tz ) - one;
            const result_type& xy    = x - y;
            const result_type& xz    = x - z;
            const result_type& yx    = y - x;
            const result_type& yz    = y - z;
            const result_type& zx    = z - x;
            const result_type& zy    = z - y;
            const result_type& ans_x = (ex_ - tx) / ( xy * xz );
            const result_type& ans_y = (ey_ - ty) / ( yx * yz );
            const result_type& ans_z = (ez_ - tz) / ( zx * zy );

            return ans_x + ans_y + ans_z;
        }

    };//struct coefficient

}//namespace f

#endif//XBWEAARGAAOAJHJCYFVFUKARCELUWRCBXFFRPGYVAIQHPNMELXAANWWVCQXIKKGDYFJCQUPTS

