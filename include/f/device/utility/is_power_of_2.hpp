#ifndef MIS_POWER_OF_2_HPP_INCLUDED_SDPOIUOJ4OUASFDIJKN4IUBADSJKFNASDFIHB4FEDFSD
#define MIS_POWER_OF_2_HPP_INCLUDED_SDPOIUOJ4OUASFDIJKN4IUBADSJKFNASDFIHB4FEDFSD

namespace device
{

    template <unsigned long X>
    struct least_significant_one_bit
    {
        static const unsigned long value= ((X ^ (X-1)) + 1) >> 1;
    };

    template <unsigned long X>
    struct is_power_of_2
    {
        static const bool value = X == least_significant_one_bit<X>::value;
    };

}//namespace device

#endif//_IS_POWER_OF_2_HPP_INCLUDED_SDPOIUOJ4OUASFDIJKN4IUBADSJKFNASDFIHB4FEDFSD

