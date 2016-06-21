#ifndef URXJHSUVITJXCIYNDPQNGFWNMVWPOCHAWHSEDRAGFHRVBVCQUJQNXCGTTQQDODVCOCFFGCKLT
#define URXJHSUVITJXCIYNDPQNGFWNMVWPOCHAWHSEDRAGFHRVBVCQUJQNXCGTTQQDODVCOCFFGCKLT

#include <f/dynamic_inverse/impl/structure_matrix.hpp>
#include <f/matrix/matrix.hpp>
#include <f/matrix/numeric/expm.hpp>

#include <f/coefficient/expm.hpp>

#include <cstddef>
#include <cassert>

#include <cstdlib>

namespace f
{
    template< typename T >
    matrix<std::complex<T>> taylor_expm( matrix<std::complex<T>> const& A )
    {
        T norm = 0.0;
        std::for_each( A.begin(), A.end(), [&norm]( std::complex<T> const& c ) { norm += ( std::real(c)*std::real(c) + std::imag(c)*std::imag(c) ); } );
        T const ratio = std::sqrt(norm) / 5.371920351148152;
        unsigned long const scaler = ratio < 1.0 ? 0 : std::ceil(std::log2(ratio));
        unsigned long const scaling_factor =  1 << scaler;
        std::complex<T> scale{ static_cast<T>(scaling_factor), 0.0 };

        auto const& A_ = A / scale;
        auto const& AA = A_ * A_;
        auto const& AAA = A_ * AA;

        matrix<std::complex<T>> I( A.row(), A.col(), std::complex<T>{ 0.0, 0.0 } );
        std::fill( I.diag_begin(), I.diag_end(), std::complex<T>{ 1.0, 0.0 } );

        auto const& P1 = AAA + std::complex<T>{ -0.205423815571221490859606, -12.65871752452031305098099 } * AA + std::complex<T>{ -58.21460179641193947200471, -3.189848964212376356715960 } * A_ + std::complex<T>{ -19.71085376106750328141397, 94.20645646169128946503649 } * I;
        auto const& P2 = AAA + std::complex<T>{ 9.410847631142442981719212, 0.0 } * AA + std::complex<T>{ 32.01029973951970099352671, 0.0 } * A_ + std::complex<T>{ 39.17363072664900708597702, 0.0 } * I;
        auto const& P3 = AAA + std::complex<T>{  -0.205423815571221490859601, 12.65871752452031305098099 } * AA + std::complex<T>{ -58.21460179641193947200470, 3.18984896421237635671600 } * A_ + std::complex<T>{ -19.71085376106750328141404, -94.20645646169128946503646 } * I;

        auto S = std::complex<T>{ 1.0/362880.0, 0.0 } * P1 * P2 * P3;

        for ( unsigned long index = 0; index != scaler; ++index )
             S *= S;

        return S;
    }

    template< typename T, typename Itor >
    matrix<std::complex<T>> const 
    make_scattering_matrix( matrix<std::size_t> const& ar, matrix<std::complex<T>> const& ug, Itor diag_begin, Itor diag_end, std::complex<T> const& thickness )
    {
        return expm( make_structure_matrix( ar, ug, diag_begin, diag_end ) * thickness );
        //return taylor_expm( make_structure_matrix( ar, ug, diag_begin, diag_end ) * thickness );
    }

    template< typename T, typename Itor >
    matrix<std::complex<T>> const 
    make_scattering_matrix( matrix<std::size_t> const& ar, matrix<T> const& ug, Itor diag_begin, Itor diag_end, std::complex<T> const& thickness )
    {
        return expm( make_structure_matrix( ar, ug, diag_begin, diag_end ) * thickness );
        //return taylor_expm( make_structure_matrix( ar, ug, diag_begin, diag_end ) * thickness );
    }

    template< typename T, typename Itor >
    matrix<std::complex<T>> const 
    make_scattering_matrix_c1( matrix<std::size_t> const& ar, matrix<std::complex<T>> const& ug, Itor diag_begin, Itor diag_end, std::complex<T> const& thickness )
    {
        return expm( make_structure_matrix( ar, ug, diag_begin, diag_end ), thickness );
    }

    template< typename T, typename Itor >
    matrix<std::complex<T>> const 
    make_scattering_matrix_c1( matrix<std::size_t> const& ar, matrix<T> const& ug, Itor diag_begin, Itor diag_end, std::complex<T> const& thickness )
    {
        return expm( make_structure_matrix( ar, ug, diag_begin, diag_end ), thickness );
    }

    template< typename T, typename Itor >
    matrix<std::complex<T>> const 
    make_scattering_matrix_c1( matrix<std::size_t> const& ar, matrix<std::complex<T>> const& ug, Itor diag_begin, Itor diag_end, std::complex<T> const& thickness, unsigned long index )
    {
        return expm( make_structure_matrix( ar, ug, diag_begin, diag_end ), thickness, index );
    }

    template< typename T, typename Itor >
    matrix<std::complex<T>> const 
    make_scattering_matrix_c1( matrix<std::size_t> const& ar, matrix<T> const& ug, Itor diag_begin, Itor diag_end, std::complex<T> const& thickness, unsigned long index )
    {
        return expm( make_structure_matrix( ar, ug, diag_begin, diag_end ), thickness, index );
    }

    template< typename T >
    void complex_produce( matrix<std::complex<T>> const& A, matrix<std::complex<T>> const& x, matrix<std::complex<T>>& b )
    {
        assert( A.col() == x.row() );
        assert( x.col() == 1 );
        assert( A.row() == b.row() );
        assert( b.col() == 1 );

        for ( std::size_t r = 0; r != A.row(); ++r )
            b[r][0] = std::inner_product( A.row_begin(r), A.row_end(r), x.begin(), std::complex<T>{0, 0} );
    }

    template< typename T >
    void compute_an( matrix<std::complex<T>> const& A, matrix<std::complex<T>> const& a_n_1, matrix<std::complex<T>>& a_n, std::size_t n )
    {
        assert( A.col() == a_n_1.row() );
        assert( a_n_1.col() == 1 );
        assert( A.row() == a_n.row() );
        assert( a_n.col() == 1 );

        T const factor = n;

        for ( std::size_t r = 0; r != A.row(); ++r )
            a_n[r][0] = std::inner_product( A.row_begin(r), A.row_end(r), a_n_1.begin(), std::complex<T>{0, 0} ) / factor;
    }

    template< typename T >
    T square_norm( matrix<std::complex<T>> const& a_n )
    {
        assert( a_n.col() == 1 );
        return std::accumulate( a_n.begin(), a_n.end(), T(0), []( T x, std::complex<T> const& c ) { return x + std::norm(c); } );
    }

    template< typename T, typename Itor >
    matrix<std::complex<T>> const 
    make_scattering_matrix( matrix<std::size_t> const& ar, matrix<std::complex<T>> const& ug, Itor diag_begin, Itor diag_end, std::complex<T> const& thickness, std::size_t column_index )
    {
        //std::cout << "debugging in make_scattering_matrix:\n";
        //std::cout << "Ar=\n" << ar << "\n";
        //std::cout << "column_index=\n" << column_index << "\n";
        assert( column_index < ar.row() );
#if 1
        matrix<std::complex<T>> const& S = make_scattering_matrix( ar, ug, diag_begin, diag_end, thickness );
        matrix<std::complex<T>> S_{ S.row(), 1 };
        std::copy( S.col_begin(column_index), S.col_end(column_index), S_.begin() );
#endif
#if 0
        typedef std::complex<T> complex_type;
        typedef matrix<std::complex<T>> complex_matrix_type;
        auto const& A = make_structure_matrix( ar, ug, diag_begin, diag_end ) * thickness;

        complex_matrix_type S_( A.row(), 1, complex_type{0.0, 0.0} );
        S_[column_index][0] = complex_type{1.0, 0.0}; 

        complex_matrix_type A_( A.row(), 1 );
        std::copy( A.col_begin(column_index), A.col_end(column_index), A_.begin() );

        S_ += A_; // I + A

        complex_matrix_type a_n_1 = A_;

        complex_matrix_type a_n = A_;

        // S_ = I + A + A^2/2 + ... + A^10/10!
        for ( std::size_t idx = 2; idx != 11; ++idx )
        {
            compute_an( A, a_n_1, a_n, idx );
            S_ += a_n;
            a_n.swap( a_n_1 );
        }

        T const eps( 1.0e-20 );

        for ( std::size_t idx = 11; true; ++idx )
        {
            compute_an( A, a_n_1, a_n, idx );
            S_ += a_n;

            T const ws = square_norm( S_ );
            T const wn = square_norm( a_n );

            if ( wn < ws * eps ) break;

            a_n.swap( a_n_1 );
        }
#endif
#if 0
        auto const& A = make_structure_matrix( ar, ug, diag_begin, diag_end ) * thickness;
        complex_matrix_type S( A.row(), A.col(), complex_type{0.0, 0.0} );

        std::fill( S.begin(), S.end(), complex_type{1.0, 0.0} );
        S += A;

        complex_matrix_type Ax = A;

        for ( std::size_t idx = 2; idx != 20; ++idx )
        {
            Ax *= A;
            Ax /= complex_type{ static_cast<T>(idx), 0 };
            S += Ax;
        }

        matrix<std::complex<T>> S_{ S.row(), 1 };
        std::copy( S.col_begin(column_index), S.col_end(column_index), S_.begin() );
#endif
#if 0

        complex_matrix_type A( ar.row(), ar.col() );
        for ( std::size_t r = 0; r != A.row(); ++r )
            for ( std::size_t c = 0; c != A.col(); ++c )
                A[r][c] = ug[ar[r][c]][0];
        std::copy( diag_begin, diag_end, A.diag_begin() );

        A *= thickness;

        //I
        complex_matrix_type S_{ A.row(), 1, std::complex<T>{0.0, 0.0} };
        std::fill( S_.begin(), S_.end(), complex_type{0.0, 0.0} );
        S_[column_index][0] = std::complex<T>{ 1.0, 0.0 };

        //A
        complex_matrix_type En{ A.row(), 1 };
        std::copy( A.col_begin( column_index ), A.col_end( column_index ), En.begin() );

        //S2
        complex_matrix_type S2{ A.row(), 1 };
        complex_produce( A, En, S2 );

        //S3
        complex_matrix_type S3{ A.row(), 1 };
        complex_produce( A, S2, S3 );

        //S4
        complex_matrix_type S4{ A.row(), 1 };
        complex_produce( A, S3, S4 );

        //S5
        complex_matrix_type S5{ A.row(), 1 };
        complex_produce( A, S4, S5 );

        //S6
        complex_matrix_type S6{ A.row(), 1 };
        complex_produce( A, S5, S6 );

        //S7
        complex_matrix_type S7{ A.row(), 1 };
        complex_produce( A, S6, S7 );

        //S8
        complex_matrix_type S8{ A.row(), 1 };
        complex_produce( A, S7, S8 );

        //S9
        complex_matrix_type S9{ A.row(), 1 };
        complex_produce( A, S8, S9 );

        //S10
        complex_matrix_type S10{ A.row(), 1 };
        complex_produce( A, S9, S10 );

        for ( std::size_t r = 0; r != S_.row(); ++r )
        {
            S_[r][0] += En[r][0] + S2[r][0] / complex_type{ 2.0, 0.0 } +
                                   S3[r][0] / complex_type{ 6.0, 0.0 } +
                                   S4[r][0] / complex_type{ 24.0, 0.0 } +
                                   S5[r][0] / complex_type{ 120.0, 0.0 } +
                                   S6[r][0] / complex_type{ 720.0, 0.0 } +
                                   S7[r][0] / complex_type{ 5040.0, 0.0 } +
                                   S8[r][0] / complex_type{ 40320.0, 0.0 } +
                                   S9[r][0] / complex_type{ 362880.0, 0.0 } +
                                   S10[r][0] / complex_type{ 3628800.0, 0.0 }; 
        }
#endif

        return S_;
    }
#if 0
    template< typename T, typename Itor >
    matrix<std::complex<T>> const 
    make_scattering_matrix( matrix<std::size_t> const& ar, matrix<T> const& ug, Itor diag_begin, Itor diag_end, std::complex<T> const& thickness, std::size_t column_index )
    {
        assert( column_index < ar.row() );
        matrix<std::complex<T>> const& S = make_scattering_matrix( ar, ug, diag_begin, diag_end, thickness );
        matrix<std::complex<T>> S_{ S.row(), 1 };
        std::copy( S.col_begin(column_index), S.col_end(column_index), S_.begin() );
        return S_;
    }
#endif

}//namespace f

#endif//URXJHSUVITJXCIYNDPQNGFWNMVWPOCHAWHSEDRAGFHRVBVCQUJQNXCGTTQQDODVCOCFFGCKLT

