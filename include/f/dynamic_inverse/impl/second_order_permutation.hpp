#ifndef DJCEDHWWYHKJFRKGYDHFVKICDPSMMVWHFYIVOLCTUXOOHPOLJFVPCIOVEDUGSLUFLMUPKUQNS
#define DJCEDHWWYHKJFRKGYDHFVKICDPSMMVWHFYIVOLCTUXOOHPOLJFVPCIOVEDUGSLUFLMUPKUQNS

#include <f/polynomial/polynomial.hpp>
#include <f/matrix/matrix.hpp>
#include <f/pattern/double_square_solver.hpp>

#include <functional>
#include <vector>
#include <algorithm>
#include <map>

#include <boost/dynamic_bitset.hpp>

namespace f
{

    template<typename T, typename Zen>
    struct second_order_permutation
    {
        typedef T                                       value_type;
        typedef std::complex<value_type>                complex_type;
        typedef Zen                                     zen_type;
        typedef unsigned long long                      size_type;
        typedef std::map<size_type, value_type>         radius_fitting_result_type;
        typedef std::map<size_type, value_type>         index_value_associate_type;
        typedef matrix<value_type>                      matrix_type;
        typedef std::vector<size_type>                  vector_type;
        typedef std::function<value_type(size_type)>    function_type;

        function_type const make_second_order_residual_function( size_type pattern_index ) const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );

            //count the unique terms -- make a vector, put id into vector, sort and unique
            vector_type vec;
            for ( size_type index = 0; index != zen.dimension(0); ++index )
            {   //ignore diagonal terms
                if ( pattern_index == index ) continue;
                if ( zen.column_index() == index ) continue;
                //load Ar[i][k] and Ar[k][j]
                size_type const A_ik_index = zen.ar( 0, pattern_index, index );
                size_type const A_kj_index = zen.ar( 0, index, zen.column_index() );
                //if all in zero_set, ignore
                if ( zen.is_zero(A_ik_index) && zen.is_zero(A_kj_index) ) continue;
                //make unique key
                size_type large = A_ik_index > A_kj_index ? A_ik_index : A_kj_index;
                size_type small = A_ik_index <= A_kj_index ? A_ik_index : A_kj_index;
                vec.push_back( (small << 32) | large );
            }
            std::sort( vec.begin(), vec.end() );
            vec.resize( std::distance( vec.begin(), std::unique( vec.begin(), vec.end() ) ) );

            //create vx/vy, note first element in vx should be for A_ij coefficients
            size_type const unknowns = vec.size()+1;
            matrix_type vx{ zen.pattern_size(), vec.size()*2+2};

            for ( size_type tilt_index = 0; tilt_index != zen.pattern_size(); ++tilt_index )
            {
                size_type const row = pattern_index;
                size_type const col = zen.column_index();
                coefficient<value_type> const coef{ zen.ipit(), zen.diag_begin(tilt_index), zen.diag_end(tilt_index) };

                //direct write A_ij coefficients
                vx[tilt_index][0] = std::real(coef(row, col));
                vx[tilt_index][unknowns] = std::real(coef(row, col));

                //two maps to cache (A_ik A_kj) coefficients, initialized incase of duplication
                index_value_associate_type real_map;
                index_value_associate_type imag_map;
                for ( auto const& key : vec )
                {
                    real_map[key] = value_type{0};
                    imag_map[key] = value_type{0};
                }
                for ( size_type index = 0; index != zen.dimension(0); ++index )
                {
                    auto c_ikj = coef( row, index, col );
                    if ( row == index || col == index )
                    {
                        c_ikj *= *(zen.diag_begin(tilt_index)+index);
                        vx[tilt_index][0] += std::real(c_ikj);
                        vx[tilt_index][unknowns] += std::imag(c_ikj);
                    }

                    size_type const A_ik_index = zen.ar( 0, pattern_index, index );
                    size_type const A_kj_index = zen.ar( 0, index, zen.column_index() );
                    if ( zen.is_zero(A_ik_index) && zen.is_zero(A_kj_index) ) continue; //ignore zero_set
                    size_type large = A_ik_index > A_kj_index ? A_ik_index : A_kj_index;
                    size_type small = A_ik_index <= A_kj_index ? A_ik_index : A_kj_index;
                    size_type const key_index = (small << 32) | large;
                    real_map[key_index] += std::real( c_ikj );
                    imag_map[key_index] += std::imag( c_ikj );
                }

                //write to vx matrix
                auto vx_real_itor = vx.row_begin(tilt_index) + 1;
                for ( auto const& elem : real_map )
                    *vx_real_itor++ = elem.second;
                auto vx_imag_itor = vx.row_begin(tilt_index) + unknowns + 1;
                for ( auto const& elem : imag_map )
                    *vx_imag_itor++ = elem.second;
            }

            matrix_type vy{ zen.pattern_size(), 1};
            for ( size_type i = 0; i != zen.pattern_size(); ++i )
                vy[i][0] = zen.intensity( i, pattern_index );

            auto const& result_function = [=vec, =vx, &zen](size_type key)
            {
                size_type const length = vec.size();
                boost::dynamic_bitset<> db( length, key );
                //TODO:
                //      generate the function
            };

        }

    };//struct second_order_permutation

}//namespace f

#endif//DJCEDHWWYHKJFRKGYDHFVKICDPSMMVWHFYIVOLCTUXOOHPOLJFVPCIOVEDUGSLUFLMUPKUQNS

