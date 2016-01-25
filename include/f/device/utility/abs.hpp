#ifndef MABS_HPP_INCLUDED_SDOINASLKJYHOIUH4KJAHBSFKAJSBFOIU4HEAKFJHBASLKJDFIUFHF
#define MABS_HPP_INCLUDED_SDOINASLKJYHOIUH4KJAHBSFKAJSBFOIU4HEAKFJHBASLKJDFIUFHF

#include <device/typedef.hpp>

namespace device
{

template <device::size_t x>
struct abs_
{
  static const device::size_t value = x < 0 ? -x : x;
};

}//namespace device

#endif//_ABS_HPP_INCLUDED_SDOINASLKJYHOIUH4KJAHBSFKAJSBFOIU4HEAKFJHBASLKJDFIUFHF

