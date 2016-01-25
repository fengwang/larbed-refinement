#ifndef LRKVXVRVBVFPQYMESQFVTNJMVVOTOOKPHFSIKMXSNMUUSHDFFGBVRQQUJECOLFWHISUVFFAMK
#define LRKVXVRVBVFPQYMESQFVTNJMVVOTOOKPHFSIKMXSNMUUSHDFFGBVRQQUJECOLFWHISUVFFAMK

#include <f/device/device_matrix/device_matrix.hpp>
#include <f/device/utility/cublas_handle.hpp>
#include <f/device/utility/value_extractor.hpp>

#include <f/singleton/singleton.hpp>

namespace f
{

    namespace norm_2_impl
    {
        template<typename T>
        struct norm_2_calculator
        {
            typedef T   value_type;
            //default here
            T operator()( device_matrix<T> const& ) const;
        };

        template<typename T>
        struct norm_2_calculator<std::complex<T>>
        {
            typedef T   value_type;
            //default complex here
            T operator()( device_matrix<T> const& ) const;
        };
#if 0
        template<>
        struct norm_2_calculator<std::complex<float>>
#endif

        template<>
        struct norm_2_calculator<std::complex<double>>
        {
            double operator()( device_matrix<std::complex<double>> const& A ) const
            {
                auto& ci = singleton<cublas_handle>::instance();

                double ans = 0.0;
                for ( unsigned long r = 0; r != A.row(); ++r )
                {
                    double amax = 0.0;
                    cublas_assert( cublasZnrm2( ci.handle, A.col(), A.data()+r*A.col(), 1, &amax ) ); 
                    if ( ans < amax ) ans = amax;
                }
                return ans;
            }
        };

    }//namespace norm_2_impl

    template<typename T>
    typename value_extractor<T>::value_type norm_2( device_matrix<T> const& A )
    {
        return norm_2_impl::norm_2_calculator<T>()( A );
    }

}//namespace f

#endif//LRKVXVRVBVFPQYMESQFVTNJMVVOTOOKPHFSIKMXSNMUUSHDFFGBVRQQUJECOLFWHISUVFFAMK

