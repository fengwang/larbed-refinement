#ifndef KFAEJYMFTFVDCIPXHGAIUAIFPSUQBFBTPKRNKRBPPXYGBOMLPGAQTGYQPPMVOTEDOWDVGBLUQ
#define KFAEJYMFTFVDCIPXHGAIUAIFPSUQBFBTPKRNKRBPPXYGBOMLPGAQTGYQPPMVOTEDOWDVGBLUQ

#include <f/matrix/impl/crtp/typedef.hpp>

#include <cassert>

namespace f
{
    template<typename Matrix, typename Type, std::size_t Default, typename Allocator>
    struct crtp_clear
    {
        typedef Matrix                                                          zen_type;
        typedef crtp_typedef<Type, Default, Allocator>                          type_proxy_type;

        void clear()
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            zen.resize(0, 0);
        }

    };//struct crtp_clear

}

#endif//KFAEJYMFTFVDCIPXHGAIUAIFPSUQBFBTPKRNKRBPPXYGBOMLPGAQTGYQPPMVOTEDOWDVGBLUQ

