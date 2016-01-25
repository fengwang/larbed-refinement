#ifndef MSQRT_HPP_INCLUDED_SDPOIN34O9HUAFLKSDJNASFKJNB34O8UABSFDKJBASFDO8HU43KJA
#define MSQRT_HPP_INCLUDED_SDPOIN34O9HUAFLKSDJNASFKJNB34O8UABSFDKJBASFDO8HU43KJA

#include <device/typedef.hpp>
#include <device/utility/abs.hpp>

namespace device
{

namespace sqrt_private
{

    template <device::size_t guess, device::size_t x, bool Converged>
    struct sqrt_impl
    {
        static device::size_t const quotient = x / guess;
        static device::size_t const new_value = (quotient + guess) / 2;
        static bool const converging = abs_<guess - quotient>::value < 2;
        static device::size_t const value = sqrt_impl<new_value, x, converging>::value;
    };

    template <device::size_t guess, device::size_t x>
    struct sqrt_impl<guess, x, true> 
    {
        static device::size_t const value = guess;
    };

}

template <device::size_t x>
struct sqrt_ 
{
      static device::size_t const value = sqrt_private::sqrt_impl<1, x, false>::value;
}; 

}//namespace device

#endif//_SQRT_HPP_INCLUDED_SDPOIN34O9HUAFLKSDJNASFKJNB34O8UABSFDKJBASFDO8HU43KJA

