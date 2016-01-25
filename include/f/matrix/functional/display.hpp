#ifndef MDISPLAY_HPP_INCLUDED_SOFDIJW4EOIJASFLD0JKSAF9LKDJSAFLKJSFDOIJSLKJASFODIJASFOIJWREOIJSAFL3JD
#define MDISPLAY_HPP_INCLUDED_SOFDIJW4EOIJASFLD0JKSAF9LKDJSAFLKJSFDOIJSLKJASFODIJASFOIJWREOIJSAFL3JD

#include <f/matrix/matrix.hpp>

#include <iostream>

namespace f
{
    template<typename T, std::size_t D, typename A>
    void display( const matrix<T,D,A>& m )
    {
        std::cout << m << std::endl;
    }

    template<typename T, std::size_t D, typename A>
    void disp( const matrix<T,D,A>& m )
    {
        display( m );
    }

}//namespace f

#endif//_DISPLAY_HPP_INCLUDED_SOFDIJW4EOIJASFLD0JKSAF9LKDJSAFLKJSFDOIJSLKJASFODIJASFOIJWREOIJSAFL3JD

