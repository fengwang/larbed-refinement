#ifndef TUQJUTPXBCXRAIPNCPGFSLUBOUFBKWPEBHEWWRVPDJARHAIGYIPEMNUOSJFXVWBKXNRIAPFVP
#define TUQJUTPXBCXRAIPNCPGFSLUBOUFBKWPEBHEWWRVPDJARHAIGYIPEMNUOSJFXVWBKXNRIAPFVP

#include <f/coefficient/coefficient.hpp>
#include <f/matrix/matrix.hpp>

#include <complex>

namespace f
{
   
    template< typename T >
    matrix<std::complex<T> > const expm( const matrix<std::complex<T> >& A, const std::complex<T>& t )
    {
        assert( A.row() == A.col() );
        unsigned long const n = A.row();

        coefficient<T> const coef( t, A.diag_begin(), A.diag_end() );

        matrix<std::complex<T> > S( n, n );

        for ( unsigned long r = 0; r != n; ++r )
        {
            T res{ 0 };
            for ( unsigned long c = 0; c != n; ++c )
            {
                std::complex<T> S_rc = A[r][c] * coef( r, c );;

                S[r][c] = S_rc;
                res += std::norm( S_rc );
            }

            S[r][r] = std::exp( t*A[r][r] ) * ( T{1} - res );
        }

        return S; 
    }
   
    template< typename T >
    matrix<std::complex<T> > const expm_2( const matrix<std::complex<T> >& A, const std::complex<T>& t )
    {
        assert( A.row() == A.col() );
        unsigned long const n = A.row();

        coefficient<T> const coef( t, A.diag_begin(), A.diag_end() );

        matrix<std::complex<T> > S( n, n );

        for ( unsigned long r = 0; r != n; ++r )
        {
            for ( unsigned long c = 0; c != n; ++c )
            {
                std::complex<T> S_rc = A[r][c] * coef( r, c );;

                for ( unsigned long l = 0; l != n; ++l )
                    S_rc += A[r][l] * A[l][c] * coef( r, l, c ); 

                S[r][c] = S_rc;
            }

            S[r][r] += std::exp( t*A[r][r] );
        }
            
        return S; 
    }
   
    //calculate the entry for column c
    template< typename T >
    matrix<std::complex<T> > const expm( const matrix<std::complex<T> >& A, const std::complex<T>& t, const unsigned long c )
    {
        assert( A.row() == A.col() );
        assert( c < A.row() );
        unsigned long const n = A.row();

        coefficient<T> const coef( t, A.diag_begin(), A.diag_end() );

        matrix<std::complex<T> > S( n, 1 );

        for ( unsigned long r = 0; r != n; ++r )
        {
                std::complex<T> S_rc = A[r][c] * coef( r, c );;
                S[r][0] = S_rc;
        }

        S[c][0] += std::exp( t*A[c][c] );
            
        return S; 
    }

    //calculate the entry for column c, C-2 coefficient
    template< typename T >
    matrix<std::complex<T> > const expm_2( const matrix<std::complex<T> >& A, const std::complex<T>& t, const unsigned long c )
    {
        assert( A.row() == A.col() );
        assert( c < A.row() );
        unsigned long const n = A.row();

        coefficient<T> const coef( t, A.diag_begin(), A.diag_end() );

        matrix<std::complex<T> > S( n, 1 );

        for ( unsigned long r = 0; r != n; ++r )
        {
                std::complex<T> S_rc = A[r][c] * coef( r, c );;

                for ( unsigned long l = 0; l != n; ++l )
                    S_rc += A[r][l] * A[l][c] * coef( r, l, c );

                S[r][0] = S_rc;
        }

        S[c][0] += std::exp( t*A[c][c] );

        return S;
    }

    template< typename T >
    std::complex<T> const expm( const matrix<std::complex<T> >& A, const std::complex<T>& t, const unsigned long r, const unsigned long c )
    {
        assert( A.row() == A.col() );
        assert( r < A.row() );
        assert( c < A.row() );

        coefficient<T> const coef( t, A.diag_begin(), A.diag_end() );

        if ( r == c ) 
            return std::exp( t * A[r][r] );

        return A[r][c] * coef( r, c );
    }

    template< typename T >
    std::complex<T> const expm_2( const matrix<std::complex<T> >& A, const std::complex<T>& t, const unsigned long r, const unsigned long c )
    {
        assert( A.row() == A.col() );
        unsigned long const n = A.row();
        assert( r < n );
        assert( c < n );

        coefficient<T> const coef( t, A.diag_begin(), A.diag_end() );

        std::complex<T> S_rc = A[r][c] * coef( r, c );;

        for ( unsigned long l = 0; l != n; ++l )
            S_rc += A[r][l] * A[l][c] * coef( r, l, c ); 

        if ( r == c ) 
            S_rc += std::exp( t*A[c][c] );
            
        return S_rc;
    }

}//namespace f

#endif//TUQJUTPXBCXRAIPNCPGFSLUBOUFBKWPEBHEWWRVPDJARHAIGYIPEMNUOSJFXVWBKXNRIAPFVP

