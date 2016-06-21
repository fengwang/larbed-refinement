#ifndef LBYSCKVMCJTNXECTJPGGJVLKSJXJPMXLKQSIGYIFSUKWGASAOQUYYOXYQHYTDWDWQYCBWBBIA
#define LBYSCKVMCJTNXECTJPGGJVLKSJXJPMXLKQSIGYIFSUKWGASAOQUYYOXYQHYTDWDWQYCBWBBIA

#include <f/dynamic_inverse/impl/c1_matrix.hpp>
#include <f/matrix/matrix.hpp>

#include <complex>
#include <cassert>
#include <cstddef>

namespace f
{
#if 0
    template< typename T >
    matrix<T> const make_double_square_coef_matrix( matrix<c1_data<T>> const& lhs_matrix, matrix<std::complex<T>> const& rhs_col_matrix, std::size_t column_index, std::size_t max_ug_index )
    {
        assert( lhs_matrix.row() == rhs_col_matrix.row() );
        assert( lhs_matrix.row() == lhs_matrix.col() );
        assert( 1 == rhs_col_matrix.col() );

        std::size_t const n = lhs_matrix.row();
        //we need to ignore the central column?

        matrix<T> ans{ n-1, max_ug_index+max_ug_index+4, T{0} };

        for ( std::size_t index = 0; index != n; ++index )
        {
            if ( index == column_index ) continue;

            std::size_t const index_se = index < column_index ? index : index - 1;
            
            for ( std::size_t jndex = 0; jndex != n; ++jndex )
            {
                std::size_t const ug_index = lhs_matrix[index][jndex].index;
                std::complex<T> const coef = lhs_matrix[index][jndex].coefficient * rhs_col_matrix[jndex][0];
                std::size_t const index_1 = ug_index == 0 ? 0 : ug_index+1;
                std::size_t const index_2 = ug_index == 0 ? max_ug_index + 2 : max_ug_index + 3 + ug_index;
                ans[index_se][index_1] = std::real( coef );
                ans[index_se][index_2] = std::imag( coef );
            }
        }
        return ans;
    }
#endif
    template< typename T >
    matrix<T> const make_double_square_coef_matrix( matrix<c1_data<T>> const& lhs_matrix, matrix<std::complex<T>> const& rhs_col_matrix, std::size_t max_ug_index )
    {
        assert( lhs_matrix.row() == rhs_col_matrix.row() );
        assert( lhs_matrix.row() == lhs_matrix.col() );
        assert( 1 == rhs_col_matrix.col() );

        std::size_t const n = lhs_matrix.row();

        matrix<T> ans{ n, max_ug_index+max_ug_index+4, T{0} };

        for ( std::size_t index = 0; index != n; ++index )
        {
            for ( std::size_t jndex = 0; jndex != n; ++jndex )
            {
                std::size_t const ug_index = lhs_matrix[index][jndex].index;
                std::complex<T> const coef = lhs_matrix[index][jndex].coefficient * rhs_col_matrix[jndex][0];
                std::size_t const index_1 = ug_index == 0 ? 0 : ug_index+1;
                std::size_t const index_2 = ug_index == 0 ? max_ug_index + 2 : max_ug_index + 3 + ug_index;
                ans[index][index_1] = std::real( coef );
                ans[index][index_2] = std::imag( coef );
            }
        }
        return ans;
    }

    //make_double_square_coef_matrix( c1_matrix, complex_matrix );

}//namespace f

#endif//LBYSCKVMCJTNXECTJPGGJVLKSJXJPMXLKQSIGYIFSUKWGASAOQUYYOXYQHYTDWDWQYCBWBBIA

