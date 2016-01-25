#ifndef XMQEVUBVWWTLRAXDBUMGAPORLBQMAAWTGOPCVNKRVHGHCMFFWAYUQHRDGBOSXMLFBMEJWLXUW
#define XMQEVUBVWWTLRAXDBUMGAPORLBQMAAWTGOPCVNKRVHGHCMFFWAYUQHRDGBOSXMLFBMEJWLXUW

#include <cuda_runtime.h>

inline float2 const operator + ( float2 const& lhs, float2 const& rhs )
{
    return float2{ lhs.x + rhs.x, lhs.y + rhs.y };
}

inline float2 const operator - ( float2 const& lhs, float2 const& rhs )
{
    return float2{ lhs.x - rhs.x, lhs.y - rhs.y };
}

inline float2 const operator * ( float2 const& lhs, float2 const& rhs )
{
    return float2{ lhs.x * rhs.x - lhs.y * rhs.y, lhs.x*lhs.y + lhs.y * rhs.x };
}

inline float abs2( float2 const& f2 )
{
    return f2.x*f2.x + f2.y*f2.y;
}

inline float2 const operator / ( float2 const& lhs, float2 const& rhs )
{
    float const c2d2 = abs2( rhs );
    return float2{ (lhs.x*rhs.x+lhs.y*rhs.y)/c2d2, (-lhs.x*rhs.y+lhs.y*rhs.x)/c2d2 };
}

bool operator < ( float2 const& lhs, float2 const& rhs )
{
    return abs2( lhs ) < abs2( rhs );
}

bool operator > ( float2 const& lhs, float2 const& rhs )
{
    return abs2( lhs ) > abs2( rhs );
}

#endif//XMQEVUBVWWTLRAXDBUMGAPORLBQMAAWTGOPCVNKRVHGHCMFFWAYUQHRDGBOSXMLFBMEJWLXUW

