#ifndef MINNER_PRODUCT_COEFFICIENT_EXTRACTOR_HPP_INCLUDED_SFDPOINASDFLKJSFDLAFKD
#define MINNER_PRODUCT_COEFFICIENT_EXTRACTOR_HPP_INCLUDED_SFDPOINASDFLKJSFDLAFKD

#include <type_traits>
#include <iterator>
#include <complex>
#include <vector>
#include <map>
#include <cstddef>
#include <cassert>

namespace f
{
    //
    // input:
    //
    // return
    //          0   ----    success
    //          1   ----    failure
    //
    template< typename Input_Itor_Symbol, typename Input_Itor_Coef_1, typename Input_Itor_Coef_2, typename Output_Itor_Coef_1, typename Output_Itor_Coef_2 >
    int extract_inner_product_coefficients( std::size_t ug_size, 
                                            std::size_t ar_dim, 
                                            Input_Itor_Symbol i_symbol, 
                                            Input_Itor_Coef_1 i_coef_1, 
                                            Input_Itor_Coef_2 i_coef_2, 
                                            Output_Itor_Coef_1 o_coef_1, 
                                            Output_Itor_Coef_2 o_coef_2 )
    {
        assert( ug_size );
        assert( ar_dim );
        /*
        assert( std::is_same<std::size_t, typename std::iterator_traits<Input_Itor_Symbol>::value_type>::value );
        assert( std::is_same<typename std::iterator_traits<Input_Itor_Coef_1>::value_type, typename std::iterator_traits<Input_Itor_Coef_2>::value_type>::value );
        assert( std::is_same<typename std::iterator_traits<Input_Itor_Coef_1>::value_type, std::complex<typename std::iterator_traits<Output_Itor_Coef_1>::value_type> >::value );
        assert( std::is_same<typename std::iterator_traits<Output_Itor_Coef_1>::value_type, typename std::iterator_traits<Output_Itor_Coef_2>::value_type>::value );
        */

        typedef std::size_t                                                         size_type;
        typedef typename std::iterator_traits<Output_Itor_Coef_1>::value_type       value_type;
        typedef typename std::iterator_traits<Input_Itor_Coef_1>::value_type        complex_type;

        std::map<size_type, value_type> o_dic_1;
        std::map<size_type, value_type> o_dic_2;
        for ( size_type index = 0; index != ug_size; ++index )
        {
            o_dic_1[index] = value_type{};
            o_dic_2[index] = value_type{};
        }

        for ( size_type step = 0; step != ar_dim; ++step )
        {
            size_type const symbol_index = *i_symbol++;
            assert( symbol_index < ug_size );
            complex_type const& the_product = *i_coef_1++ * *i_coef_2++;
            value_type const the_real = std::real( the_product );
            value_type const the_imag = std::imag( the_product );
            o_dic_1[symbol_index] += the_real;
            o_dic_2[symbol_index] += the_imag;
        }

        for ( size_type index = 0; index != ug_size; ++index )
        {
            *o_coef_1++ = o_dic_1[index];
            *o_coef_2++ = o_dic_2[index];
        }

        return 0;
    }

}//namespace f

#endif//_INNER_PRODUCT_COEFFICIENT_EXTRACTOR_HPP_INCLUDED_SFDPOINASDFLKJSFDLAFKD

