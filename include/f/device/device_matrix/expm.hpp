#ifndef YBCGURXOCLUFHOQCWTFWVWMRYWMGHNXMACFEEOCBASDQOQFWAMCYNFBLAYAPNJLAKBCPYAWAS
#define YBCGURXOCLUFHOQCWTFWVWMRYWMGHNXMACFEEOCBASDQOQFWAMCYNFBLAYAPNJLAKBCPYAWAS

#include <expm/device_matrix.hpp>

namespace f
{
    c_matrix const expm( const c_matrix& A );
    extern "C" void expm( const std::complex<float>* A, unsigned long n, std::complex<float>* M );

#if 0
    c_matrix const expm( const c_matrix& A )
    {
        assert( A.row() == A.col() );
        const unsigned long n = A.row();
        const float nm = A.norm();
        const unsigned long s = nm < 1.0 ? 0 : static_cast<unsigned long>(std::ceil(std::log2(nm)));
        const float s_2 = s ? (1 << s) : 1; 

        /*
        const c_matrix& a = A / s_2; //the scaled matrix A

        c_matrix a_n = a; // keeps a^n
        c_matrix ans = eye_c_matrix(n); //the ans of e^A
        const unsigned long loops = 10;
        float factor = 1.0;
        for ( unsigned long i = 1; i != loops; ++i )
        {
            factor /= i;
            ans = ans + a_n * factor;
            a_n = a_n * a;
        }
        */
        //the parameters to be used for 9-orders
/*
   0.963405714485326        0.583581039167404           0.144651158571306           0.0134226317770268      
   1.01197445118178         0.335669463894947           0.0857542043420956          0.0121745404670141      
   1.0257020561269          0.0641622038836256          0.00569162022190139         0.00728977857272476 
   */
//        const float c0[4] = { 0.963405714485326, 0.583581039167404, 0.144651158571306, 0.0134226317770268 };
//        const float c1[4] = { 1.01197445118178, 0.335669463894947, 0.0857542043420956, 0.0121745404670141 };
//        const float c2[4] = { 1.0257020561269, 0.0641622038836256, 0.00569162022190139, 0.00728977857272476 };
//
/*
1.0476935979303396707
0.13621658523122884277
0.17503285846386393287
0.0001955881771752682294
0.91452063342352907238
0.67099326513851953457
0.17186329865997720945
0.00019137545763237996514
1.043692545657082027
0.13755748432914494117
-0.067225888057316357527
0.023963840848021274871
*/
        //const float c0[4] = { 1.0476935979303396707, 0.13621658523122884277, 0.17503285846386393287, 0.0001955881771752682294 };
        //const float c1[4] = { 0.91452063342352907238, 0.67099326513851953457, 0.17186329865997720945, 0.00019137545763237996514 };
        //const float c2[4] = { 1.043692545657082027, 0.13755748432914494117, -0.067225888057316357527, 0.023963840848021274871 };

        /*
0.73769924933048902638
0.13172525137117249705
0.072080085802694884856
-2.1299489346443743676e-06
-0.77307934030428138161
-0.76302801814514820133
-0.28579955930369160821
-0.041199562184697238743
-1.7534629347707793023
0.29030564199814418158
-0.086501686893504473042
3.1635840353521517065e-06
        */

        const float c0[4] = { 0.73769924933048902638, 0.13172525137117249705, 0.072080085802694884856, -2.1299489346443743676e-06 };
        const float c1[4] = { -0.77307934030428138161, -0.76302801814514820133, -0.28579955930369160821, -0.041199562184697238743 };
        const float c2[4] = { -1.7534629347707793023, 0.29030564199814418158, -0.086501686893504473042, 3.1635840353521517065e-06 };

        auto const& I = eye_c_matrix(n);
        auto const& a = A / s_2;
        auto const& aa = a*a;
        auto const& aaa = a * aa;

        auto const& P0 = c0[0] * I + c0[1] * a + c0[2] * aa + c0[3] * aaa;
        auto const& P1 = c1[0] * I + c1[1] * a + c1[2] * aa + c1[3] * aaa;
        auto const& P2 = c2[0] * I + c2[1] * a + c2[2] * aa + c2[3] * aaa;

        auto ans = P0 * P1 * P2;

        for ( unsigned long i = 0; i != s; ++i ) //squaring back
        {
            ans = ans * ans;
        }

        return ans;    
    }

    // M = e^A
    void expm( const std::complex<float>* A, unsigned long n, std::complex<float>* M )
    {
        float* a_ = new float[ 2*n*n ];
        for ( unsigned long i = 0; i != n*n; ++i )
        {
            a_[i+i]     = A[i].real();
            a_[i+i+1]   = A[i].imag();
        }

        const c_matrix a( n, n, a_ ); //the matrix A constructed from host memory

        auto const& ans = expm( a );

        ans.export_to( a_ );    

        for ( unsigned long i = 0; i != n*n; ++i )
            M[i] = std::complex<float>( a_[i+i], a_[i+i+1] );

        delete[] a_;
    }
#endif

}//namespace f

#endif//YBCGURXOCLUFHOQCWTFWVWMRYWMGHNXMACFEEOCBASDQOQFWAMCYNFBLAYAPNJLAKBCPYAWAS
