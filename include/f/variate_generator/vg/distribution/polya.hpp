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
#ifndef MPOLYA_HPP_INCLUDED_SDOIU39ASFD98UY348OAUYSFD98Y34897ASFDOI348OUYAFSDIKH34IOUAFSDOIU349Y8ASFDKLJ
#define MPOLYA_HPP_INCLUDED_SDOIU39ASFD98UY348OAUYSFD98Y34897ASFDOI348OUYAFSDIKH34IOUAFSDOIU349Y8ASFDKLJ 

#include <f/variate_generator/vg/distribution/negative_binomial.hpp>
#include <f/singleton/singleton.hpp>

#include <cassert>
#include <cstddef>
#include <cmath>

namespace f
{

    template < typename Return_Type, typename Engine >
    struct polya :  private negative_binomial<Return_Type, Engine>
    {
            typedef negative_binomial<Return_Type, Engine>     negative_binomial_type;
            typedef typename Engine::final_type                final_type;
            typedef Return_Type                                return_type;
            typedef Return_Type                                value_type;
            typedef typename Engine::seed_type                 seed_type;
            typedef std::size_t                                size_type;
            typedef Engine                                     engine_type;

            size_type           n_;
            final_type          p_;
            engine_type&        e_;

            explicit polya(     size_type n = size_type( 1 ), final_type p = final_type( 0.5 ), const seed_type s = seed_type( 0 ) )
                : n_( n ), p_( p ), e_( singleton<engine_type>::instance() )
            {
                assert( n != size_type( 0 ) );
                assert( p > final_type( 0 ) );
                assert( p < final_type( 1 ) );
                e_.reset_seed( s );
            }

            return_type
            operator()() const
            {
                return do_generation( n_, p_ );
            }

        protected:
            return_type
            do_generation( const size_type N, const final_type P ) const
            {
                return direct_impl( N, P );
            }

        private:
            return_type
            direct_impl( const size_type N, const final_type P ) const
            {
                const final_type ans = negative_binomial_type::do_generation( N, P );
                return ans;
            }
    };

}//namespace f

#endif//_POLYA_HPP_INCLUDED_SDOIU39ASFD98UY348OAUYSFD98Y34897ASFDOI348OUYAFSDIKH34IOUAFSDOIU349Y8ASFDKLJ

