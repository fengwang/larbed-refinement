#ifndef BPNYWRDNILTHNBLUXRFDOHLRQVMQOJQNWAAMKHIAPJDUKDYTTBTQMJFNRNXJHVFIVKDSPIVDX
#define BPNYWRDNILTHNBLUXRFDOHLRQVMQOJQNWAAMKHIAPJDUKDYTTBTQMJFNRNXJHVFIVKDSPIVDX

#include <f/pattern/pattern.hpp>
#include <f/coefficient/coefficient.hpp>
#include <f/coefficient/expm.hpp>
#include <f/dynamic_inverse/impl/structure_matrix.hpp>
#include <f/dynamic_inverse/impl/scattering_matrix.hpp>
#include <f/algorithm/for_each.hpp>

#include <functional>
#include <vector>
#include <cassert>

namespace f
{
    
    template< typename T >
    struct adaptive_sd
    {
        typedef T                                       value_type;
        typedef std::complex<T>                         complex_type;
        typedef value_type*                             pointer;
        typedef matrix<value_type>                      matrix_type;
        typedef matrix<complex_type>                    complex_matrix_type;
        typedef std::vector<matrix_type>                matrix_vector_type;
        typedef std::vector<complex_matrix_type>        complex_matrix_vector_type;
        typedef std::size_t                             size_type;
        typedef std::vector<size_type>                  size_vector_type;
        
        pattern<value_type>const&               pt;
        value_type                              alpha;
        value_type                              ac;
        value_type                              dc;
        value_type                              thickness;
        complex_matrix_type                     ug;

        std::function<value_type(pointer)> make_merit_function()
        {
            //proposed data structure
            //--(ac)(dc)(1r)(1i)......(Nr)(Ni)(thickness)
            return [this]( pointer p )
            {
                (*this).update_ac( p );
                (*this).update_dc( p );
                (*this).update_thickness( p );
                (*this).update_ug( p );

                return (*this).make_diff_2( (*this).ac, (*this).dc, (*this).thickness, (*this).ug );
            };
        }

        void config_alpha( value_type alpha_ )
        {
            alpha = alpha_;
        }

        adaptive_sd( pattern<value_type> const& pt_, value_type alpha_ ) : pt( pt_ ), alpha( alpha_ ) 
        {
            ug_cache.resize( pt.ug_size, 1 );
        }

    };//struct adaptive_sd

}//namespace f

#endif//BPNYWRDNILTHNBLUXRFDOHLRQVMQOJQNWAAMKHIAPJDUKDYTTBTQMJFNRNXJHVFIVKDSPIVDX

