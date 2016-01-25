#ifndef MINVERSE_HPP_INCLUDED_DSOIJ498UASFDKLJHVJNSAFIUH49UHSFDKJHSJKNMVCKJHAFDSKJHAUIHASDKJVKJHSFUH
#define MINVERSE_HPP_INCLUDED_DSOIJ498UASFDKLJHVJNSAFIUH49UHSFDKJHSJKNMVCKJHAFDSKJHAUIHASDKJVKJHSFUH

#include <f/matrix/matrix.hpp>

#include <algorithm>
#include <cstddef>

namespace f
{
    template<typename Matrix>
    Matrix const inverse( const Matrix& m )
    {
        return m.inverse();
    }

    template<typename Matrix>
    Matrix const inv( const Matrix& m )
    {
        return m.inverse();
    }

}//namespace f

#endif//_INVERSE_HPP_INCLUDED_DSOIJ498UASFDKLJHVJNSAFIUH49UHSFDKJHSJKNMVCKJHAFDSKJHAUIHASDKJVKJHSFUH
