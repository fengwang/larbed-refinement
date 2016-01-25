#ifndef MMAX_HPP_INCLUDED_SFDOJI4938YUALFKJALKHDSFIUHVCJSBANFDKIJUH4EU7HFDSKJHASIUH4E87UGHFASIDUHASFI
#define MMAX_HPP_INCLUDED_SFDOJI4938YUALFKJALKHDSFIUHVCJSBANFDKIJUH4EU7HFDSKJHASIUH4E87UGHFASIDUHASFI

#include <f/matrix/matrix.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>

namespace f
{
    template< typename T, std::size_t D, typename A>
    T const 
    max( const matrix<T,D,A>& m )
    {
        assert( m.size() );

        return *std::max_element( m.begin(), m.end() );
    }

}//namespace f

#endif//_MAX_HPP_INCLUDED_SFDOJI4938YUALFKJALKHDSFIUHVCJSBANFDKIJUH4EU7HFDSKJHASIUH4E87UGHFASIDUHASFI

