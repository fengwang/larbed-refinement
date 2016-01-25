#ifndef MASSERT_HPP_INCLUDED_SDFOIASDF984HAKJFNASKDJN398HAFLKDJASFDKJH498HFSD4FS
#define MASSERT_HPP_INCLUDED_SDFOIASDF984HAKJFNASKDJN398HAFLKDJASFDKJH498HFSD4FS

#ifdef assert
#undef assert
#endif

#ifdef NDEBUG
    #define assert(e) (static_cast<void>(0))
#else
    extern "C" int printf(const char * __restrict, ...);
    #define assert(e) (  ((e) ?  static_cast<void>(0) : ((void)printf("%s:%u: failed assertion `%s'\n", __FILE__, __LINE__, e), abort()) ) )
#endif//NDEBUG

#endif//_ASSERT_HPP_INCLUDED_SDFOIASDF984HAKJFNASKDJN398HAFLKDJASFDKJH498HFSD4FS

