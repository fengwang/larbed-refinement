/*
vg -- random variate generator library 
Copyright (C) 2010-2012  Feng Wang (feng.wang@uni-ulm.de) 

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by 
the Free Software Foundation, either version 3 of the License, or 
(at your option) any later version. 

This program is distributed in the hope that it will be useful, 
but WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
GNU General Public License for more details. 

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef MNEGATIVE_BINOMIAL_BETA_HPP_INCLUDED_SDOU347UJFDHD87Y4KUYASD786Y4KAUYSFD9784ALFKSHQ2IOPUYASDKFJLH4RUIHAFSD
#define MNEGATIVE_BINOMIAL_BETA_HPP_INCLUDED_SDOU347UJFDHD87Y4KUYASD786Y4KAUYSFD9784ALFKSHQ2IOPUYASDKFJLH4RUIHAFSD 

#include <f/variate_generator/vg/distribution/generalized_hypergeometric_b3.hpp>
#include <f/singleton/singleton.hpp>

#include <cassert>

namespace f
{

    template < typename Return_Type, typename Engine >
    struct negative_binomial_beta : private generalized_hypergeometric_b3<Return_Type, Engine>
    {
        typedef generalized_hypergeometric_b3< Return_Type, Engine> GHgB3_type;
        typedef typename Engine::final_type     final_type;
        typedef Return_Type                     return_type;
        typedef Return_Type                     value_type;
        typedef typename Engine::seed_type      seed_type;
        typedef std::size_t                     size_type;
        typedef Engine                          engine_type;

        value_type           a_;   
        value_type           b_;  
        value_type           c_; 
        engine_type&         e_;

        explicit negative_binomial_beta( value_type a = 1, value_type b = 1, value_type c = 1 , seed_type s = 0 )
                : a_( a ), b_( b ), c_( c ), e_( singleton<engine_type>::instance() ) 
        {
            assert( a > 0 );
            assert( b > 0 );
            assert( c > 0 );
            e_.reset_seed( s );
        }

        return_type
        operator()() const
        {
            return do_generation( a_, b_, c_ );
        }

    protected:
        return_type
        do_generation( const value_type a, const value_type b, const value_type c ) const
        {
            return direct_impl( a, b, c );
        }

    private:
        return_type
        direct_impl( const value_type a, const value_type b, const value_type c ) const
        {
            const final_type ans = GHgB3_type::do_generation( a, b, c );
            return ans;
        }
    };

}//namespace f

#endif//_NEGATIVE_BINOMIAL_BETA_HPP_INCLUDED_SDOU347UJFDHD87Y4KUYASD786Y4KAUYSFD9784ALFKSHQ2IOPUYASDKFJLH4RUIHAFSD 

