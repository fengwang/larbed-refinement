#ifndef IOAWQPANFNKGGLWFUNDSKCXVLYFEBTBRYGCSBMSNNNBMSFYFWGEOFEPQOSUXNMTQYWFSRRFPQ
#define IOAWQPANFNKGGLWFUNDSKCXVLYFEBTBRYGCSBMSNNNBMSFYFWGEOFEPQOSUXNMTQYWFSRRFPQ

#include <f/device/assert/cublas_assert.hpp>
#include <f/device/device_matrix/numeric/eye.hpp>
#include <f/device/device_matrix/numeric/norm.hpp>
#include <f/device/utility/cublas_handle.hpp>
#include <f/singleton/singleton.hpp>

#include <complex>
#include <cmath>
#include <cassert>

//to calculate 
// P = alpha_0 P0 + alpha_1 P1 + alpha_2 P2 + alpha_3 P3 
// implemented in file 'src/device/kernel/device_matrix/zmatrix_poly4_eval.cu'
void zmatrix_poly4_eval( unsigned long n, double2* P, double alpha_0, double2* P0, double alpha_1, double2* P1, double alpha_2, double2* P2, double alpha_3, double2* P3 );
// P = P0 + alpha_1 P1 + alpha_2 P2 + alpha_3 P3
// implemented in file 'src/device/kernel/device_matrix/zmatrix_poly4_eval.cu'
void zmatrix_poly4_eval( unsigned long n, double2* P, double2* P0, double alpha_1, double2* P1, double alpha_2, double2* P2, double alpha_3, double2* P3 );

//x^3  + 4.6027494462684224740287038180501168 x^2  + 4.3846445331453979503692027283066828 I x^2  + 15.611949003840816783610314223770969 x + 18.510564381485517241363978456937948 I x
//     + 5.3361950563949313409201792145852824 + 61.399028939445575913258464907489851 I
// P = P0 + (alpha_1_r, alpha_1_i) P1 + (alpha_2_r, alpha_2_i) P2 + (alpha_3_r, alpha_3_i) P3
void zmatrix_poly4_eval( unsigned long n, double2* P, double2* P0, double alpha_1_r, double alpha_1_i, double2* P1, double alpha_2_r, double alpha_2_i, double2* P2, double alpha_3_r, double alpha_3_i, double2* P3 );




namespace f
{

    namespace expm_impl
    {
        
        template< typename T >
        struct expm_calculator;

        template<> 
        struct expm_calculator<std::complex<double>>
        {
            typedef std::complex<double>                    complex_type;
            typedef device_matrix<std::complex<double>>     matrix_type;
            matrix_type const operator()( matrix_type const& A, unsigned long const orders ) const
            {
                assert( A.row() == A.col() );

                unsigned long const n = A.row();
                //double const nm       = norm( A );
                double const nm       = A.norm()/n;
                unsigned long const s = nm < 1.0 ? 0 : static_cast<unsigned long>(std::ceil( std::log2(nm) ) );
                double const s_2      = s ? ( 1<<s ) : 1;
                matrix_type const& a  = A / s_2;

#if 0
        const double c0[4] = { 0.73769924933048902638, 0.13172525137117249705, 0.072080085802694884856, -2.1299489346443743676e-06 };
        const double c1[4] = { -0.77307934030428138161, -0.76302801814514820133, -0.28579955930369160821, -0.041199562184697238743 };
        const double c2[4] = { -1.7534629347707793023, 0.29030564199814418158, -0.086501686893504473042, 3.1635840353521517065e-06 };

        auto const& I = device_eye<complex_type>( n );
        auto const& aa = a*a;
        auto const& aaa = a * aa;

        matrix_type P0( A.row(), A.col() );
        matrix_type P1( A.row(), A.col() );
        matrix_type P2( A.row(), A.col() );

        zmatrix_poly4_eval( A.size(), P0.data(), c0[0], I.data(), c0[1], a.data(), c0[2], aa.data(), c0[3], aaa.data() );
        zmatrix_poly4_eval( A.size(), P1.data(), c1[0], I.data(), c1[1], a.data(), c1[2], aa.data(), c1[3], aaa.data() );
        zmatrix_poly4_eval( A.size(), P2.data(), c2[0], I.data(), c2[1], a.data(), c2[2], aa.data(), c2[3], aaa.data() );
/*
        auto const& P0 = c0[0] * I + c0[1] * a + c0[2] * aa + c0[3] * aaa;
        auto const& P1 = c1[0] * I + c1[1] * a + c1[2] * aa + c1[3] * aaa;
        auto const& P2 = c2[0] * I + c2[1] * a + c2[2] * aa + c2[3] * aaa;
*/
        auto ans = P0 * P1 * P2;
#endif

#if 1
                matrix_type a_n       = a;
                matrix_type ans       = device_eye<complex_type>( n );
                double factor         = 1.0;

                for ( unsigned long i = 1; i != orders; ++i )
                {
                    factor /= i;
                    ans     = ans + a_n * factor;
                    
                    if ( i+1 == orders ) break;

                    a_n     = a_n * a;
                }
#endif

                for ( unsigned long i = 0; i != s; ++i )
                    ans = ans * ans;

                return ans;
            }
        };
    
    }//namespace expm_impl

    template< typename T >
    device_matrix<T> const expm( device_matrix<T> const& A, unsigned long const orders = 9 )
    {
        return expm_impl::expm_calculator<T>()( A, orders );
    }

    device_matrix<std::complex<double>> const expm_9( device_matrix<std::complex<double>> const& A )
    {
        typedef std::complex<double>                complex_type;
        typedef device_matrix<complex_type>         matrix_type;
        assert( A.row() == A.col() );

        unsigned long const n   = A.row();
        double const nm         = A.norm() / n;
        unsigned long const s   = nm < 1.0 ? 0 : static_cast<unsigned long>(std::ceil( std::log2(nm) ) );
        double const s_2        = s ? ( 1<<s ) : 1;
        matrix_type const& a    = A / s_2;
        matrix_type const& aa   = a*a;
        matrix_type const& aaa  = aa*a;
        matrix_type I           = device_eye<complex_type>( n );

        matrix_type P0( n, n );
        matrix_type P1( n, n );
        matrix_type P2( n, n );

        //x^3  + 9.4108476311424429817192114293198773 x^2  + 32.010299739519700993526703045242105 x + 39.173630726649007085977022185666594        
        //
        //zmatrix_poly4_eval( unsigned long n, double2* P, double2* P0, double alpha_1, double2* P1, double alpha_2, double2* P2, double alpha_3, double2* P3 );
        zmatrix_poly4_eval( n*n, P0.data(),  aaa.data(), 
                            9.4108476311424429817192114293198773, aa.data(), 
                            32.010299739519700993526703045242105, a.data(), 
                            39.173630726649007085977022185666594, I.data() );

        //x^3  + 4.6027494462684224740287038180501168 x^2  + 4.3846445331453979503692027283066828 I x^2  + 15.611949003840816783610314223770969 x + 18.510564381485517241363978456937948 I x
        //     + 5.3361950563949313409201792145852824 + 61.399028939445575913258464907489851 I
        //
        zmatrix_poly4_eval( n*n, P1.data(), aaa.data(),
                            4.6027494462684224740287038180501168, 4.3846445331453979503692027283066828, aa.data(),
                            15.611949003840816783610314223770969, 18.510564381485517241363978456937948, a.data(),
                            5.3361950563949313409201792145852824, 61.399028939445575913258464907489851, I.data() );
        
        //
        //x^3  - 5.0135970774108654557479152473699942 x^2  - 4.3846445331453979503692027283066828 I x^2  + 32.095399202760843419253361350193330 x + 23.653696832396624699846938534814053 I x
        //     + 13.013971209325298438705971137884319 - 149.74062725478794246266244284870375 I
        zmatrix_poly4_eval( n*n, P2.data(), aaa.data(),
                           -5.0135970774108654557479152473699942, -4.3846445331453979503692027283066828, aa.data(),
                            32.095399202760843419253361350193330,  23.653696832396624699846938534814053, a.data(),
                            13.013971209325298438705971137884319, -149.74062725478794246266244284870375, I.data() );
        
        auto& ans = I;
        ans = P0 * P1 * P2 / 362880.0;

        for ( unsigned long i = 0; i != s; ++i )
            ans = ans * ans;

        return ans;
    }

    device_matrix<std::complex<double>> const expm_12( device_matrix<std::complex<double>> const& A )
    {
        typedef std::complex<double>                complex_type;
        typedef device_matrix<complex_type>         matrix_type;
        assert( A.row() == A.col() );

        unsigned long const n   = A.row();
        double const nm         = A.norm() / n;
        unsigned long const s   = nm < 1.0 ? 0 : static_cast<unsigned long>(std::ceil( std::log2(nm) ) );
        double const s_2        = s ? ( 1<<s ) : 1;
        matrix_type const& a    = A / s_2;
        matrix_type const& aa   = a*a;
        matrix_type const& aaa  = aa*a;
        matrix_type I           = device_eye<complex_type>( n );

        matrix_type P0( n, n );
        matrix_type P1( n, n );
        matrix_type P2( n, n );
        matrix_type P3( n, n );
#if 0
        \left( {x}^{3}+ 11.960113580975467623845040806681209\,{x}^{2}+
                2.3027571492470953276351045107744921\,i{x}^{2}+
                48.216387479963393384471191061785405\,x+
                 19.046602878903642636325578024029306\,ix+
                 65.310893917618476265868251033357330+
                 40.769672808922641940108972832504098\,i \right)
#endif
        zmatrix_poly4_eval( n*n, P0.data(), aaa.data(),
                11.960113580975467623845040806681209, 2.3027571492470953276351045107744921, aa.data(),
                48.216387479963393384471191061785405, 19.046602878903642636325578024029306, a.data(),
                65.310893917618476265868251033357330, 40.769672808922641940108972832504098, I.data() );

#if 0
        \left( {x}^{3}+
                 9.2048755648300476498938236615244651\,{x}^{2}-
                 2.3027571492470953276351045107744921\,i{x}^{2}+
                 42.036000928319459605557351021467572\,x-
                 12.701958839347392927389871045441117\,ix+
                  80.005257638245259295967979727473762-
                  49.942482505585608164471982697687926\,i \right)
#endif
        zmatrix_poly4_eval( n*n, P1.data(), aaa.data(),
                9.2048755648300476498938236615244651, -2.3027571492470953276351045107744921, aa.data(),
                42.036000928319459605557351021467572, -12.701958839347392927389871045441117, a.data(),
                80.005257638245259295967979727473762, -49.942482505585608164471982697687926, I.data() );

#if 0
        \left( {x}^{3}+
                1.4448266470605688180137700016505612\,{x}^{2}+
                 6.0594914338643727910205790817884843\,i{x}^{2}+
                 24.426604495463644394877697266203611\,x+
                 15.138126212009885920288795371008840\,ix-
                 28.503874364304155245311580398456749+
                  163.95966071048634009892790465937710\,i \right)
#endif
        zmatrix_poly4_eval( n*n, P2.data(), aaa.data(),
                1.4448266470605688180137700016505612, 6.0594914338643727910205790817884843, aa.data(),
                24.426604495463644394877697266203611, 15.138126212009885920288795371008840, a.data(),
                -28.503874364304155245311580398456749, 163.95966071048634009892790465937710, I.data() );
#if 0
 \left( {x}^{3}-
         10.609815792866084091752634469856235\,{x}^{2}-
         6.0594914338643727910205790817884843\,i{x}^{2}+
          74.515763286992166021758598193373921\,x+
          57.906876391023589465359778838904725\,ix-
          67.891925160606028223684424127967459-
          390.52715683643666259132465969063316\,i \right)
#endif
        zmatrix_poly4_eval( n*n, P3.data(), aaa.data(),
                 10.609815792866084091752634469856235, -6.0594914338643727910205790817884843, aa.data(),
                 74.515763286992166021758598193373921,  57.906876391023589465359778838904725, a.data(),
                -67.891925160606028223684424127967459, -390.52715683643666259132465969063316, I.data() );
        
        auto& ans = I;
        ans = P0 * P1 * P2 *P3 / 479001600.0;

        for ( unsigned long i = 0; i != s; ++i )
            ans = ans * ans;

        return ans;
    }

}//namespace f

#endif//IOAWQPANFNKGGLWFUNDSKCXVLYFEBTBRYGCSBMSNNNBMSFYFWGEOFEPQOSUXNMTQYWFSRRFPQ

