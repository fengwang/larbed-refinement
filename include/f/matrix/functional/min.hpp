#ifndef MMIN_HPP_INCLUDED_SFDOJI4938YUALFKJALKHDSFIUHVCJSBANFDKIJUH4EU7HFDSKJHASIUH4E87UGHFASIDUHASFI
#define MMIN_HPP_INCLUDED_SFDOJI4938YUALFKJALKHDSFIUHVCJSBANFDKIJUH4EU7HFDSKJHASIUH4E87UGHFASIDUHASFI

#include <f/matrix/matrix.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>

namespace f
{
    template< typename T, std::size_t D, typename A>
    T const 
    min( const matrix<T,D,A>& m )
    {
        assert( m.size() );

        return std::min_element( m.begin(), m.end() );
    }

}//namespace f

#endif//_MIN_HPP_INCLUDED_SFDOJI4938YUALFKJALKHDSFIUHVCJSBANFDKIJUH4EU7HFDSKJHASIUH4E87UGHFASIDUHASFI

