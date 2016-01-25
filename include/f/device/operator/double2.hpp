#ifndef MDOUBLE_2_HPP_INCLUDED_FDSPIOASLDKJSDAL3PINASLFJKASD3UHASFDLKJHASKLFSDIU
#define MDOUBLE_2_HPP_INCLUDED_FDSPIOASLDKJSDAL3PINASLFJKASD3UHASFDLKJHASKLFSDIU

#include <cuda_runtime.h>

inline double2 const operator + ( double2 const& lhs, double2 const& rhs )
{
    return double2{ lhs.x + rhs.x, lhs.y + rhs.y };
}

inline double2 const operator - ( double2 const& lhs, double2 const& rhs )
{
    return double2{ lhs.x - rhs.x, lhs.y - rhs.y };
}

inline double2 const operator * ( double2 const& lhs, double2 const& rhs )
{
    return double2{ lhs.x * rhs.x - lhs.y * rhs.y, lhs.x*lhs.y + lhs.y * rhs.x };
}

inline float abs2( double2 const& f2 )
{
    return f2.x*f2.x + f2.y*f2.y;
}

inline double2 const operator / ( double2 const& lhs, double2 const& rhs )
{
    float const c2d2 = abs2( rhs );
    return double2{ (lhs.x*rhs.x+lhs.y*rhs.y)/c2d2, (-lhs.x*rhs.y+lhs.y*rhs.x)/c2d2 };
}

bool operator < ( double2 const& lhs, double2 const& rhs )
{
    return abs2( lhs ) < abs2( rhs );
}

bool operator > ( double2 const& lhs, double2 const& rhs )
{
    return abs2( lhs ) > abs2( rhs );
}

#endif//_DOUBLE_2_HPP_INCLUDED_FDSPIOASLDKJSDAL3PINASLFJKASD3UHASFDLKJHASKLFSDIU


