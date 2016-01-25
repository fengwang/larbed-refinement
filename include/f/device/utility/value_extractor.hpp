#ifndef UDBOMPPSKHABPCFLXOULDRKIAGUGRMPPSLOTQUYOBTOMAQVCSOFVWYWEJOARYTTUFGMDBNOVV
#define UDBOMPPSKHABPCFLXOULDRKIAGUGRMPPSLOTQUYOBTOMAQVCSOFVWYWEJOARYTTUFGMDBNOVV

#include <type_traits>
#include <complex>

namespace f
{

namespace value_extractor_private
{

    template< typename T >
    struct the_value_extractor
    {
        typedef T       value_type;
    };

    template< typename T >
    struct the_value_extractor< std::complex<T> >
    {
        typedef T       value_type;
    };

}//namespace value_extractor_private

template< typename T >
struct value_extractor
{
    typedef typename std::remove_reference<T>::type                                     type0;
    typedef typename std::remove_pointer<type0>::type                                   type1;
    typedef typename std::remove_cv<type1>::type                                        type;
    typedef typename value_extractor_private::the_value_extractor<type>::value_type     value_type;
};

}//namespace f

#endif//UDBOMPPSKHABPCFLXOULDRKIAGUGRMPPSLOTQUYOBTOMAQVCSOFVWYWEJOARYTTUFGMDBNOVV

