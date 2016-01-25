#include <f/device/device_assert/cuda_assert.hpp>
#include <f/device/device_assert/cublas_assert.hpp>
#include <f/device/device_assert/kernel_assert.hpp>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cuComplex.h>
#include <math_functions.h>

#if 1
//should call with Dznrm2<<<1,128>>>(...)
__global__ void Dznrm2( unsigned long m, double2 *dA, double *dxnorm )
{
    unsigned long i = threadIdx.x;

    __shared__ double x[128];

    double lsum = 0.0;

    for( unsigned long j = i; j < m; j += 128 ) 
    {
        double const re = dA[j].x;
        double const im = dA[j].y;
        lsum += re*re + im*im;
    }

    x[i] = lsum;

    __syncthreads();

    if ( i <   64 ) { x[i] += x[i+  64]; }  __syncthreads();
    if ( i <   32 ) { x[i] += x[i+  32]; }  __syncthreads();
    if ( i <   16 ) { x[i] += x[i+  16]; }  __syncthreads();
    if ( i <    8 ) { x[i] += x[i+   8]; }  __syncthreads();
    if ( i <    4 ) { x[i] += x[i+   4]; }  __syncthreads();
    if ( i <    2 ) { x[i] += x[i+   2]; }  __syncthreads();
    if ( i <    1 ) { x[i] += x[i+   1]; }  __syncthreads();

    if ( i == 0 ) *dxnorm = sqrt(x[0]);
}

__global__ void Dasum( unsigned long m, double2 *dA, double *dxnorm )
{
    unsigned long i = threadIdx.x;

    __shared__ double x[128];

    double lsum = 0.0;

    for( unsigned long j = i; j < m; j += 128 ) 
    {
        double const re = dA[j].x;
        double const im = dA[j].y;
        lsum += sqrt(re*re + im*im);
    }

    x[i] = lsum;

    __syncthreads();

    if ( i <   64 ) { x[i] += x[i+  64]; }  __syncthreads();
    if ( i <   32 ) { x[i] += x[i+  32]; }  __syncthreads();
    if ( i <   16 ) { x[i] += x[i+  16]; }  __syncthreads();
    if ( i <    8 ) { x[i] += x[i+   8]; }  __syncthreads();
    if ( i <    4 ) { x[i] += x[i+   4]; }  __syncthreads();
    if ( i <    2 ) { x[i] += x[i+   2]; }  __syncthreads();
    if ( i <    1 ) { x[i] += x[i+   1]; }  __syncthreads();

    if ( i == 0 ) *dxnorm = x[0];
}
#endif

#if 0
__global__ void Dznrm2( unsigned long int n, double2* x, double* the_norm )
{
    __shared__ double sSum[512];

    double res = 0.0;

    double2* lastX = x + n;

    x += threadIdx.x + blockIdx.x*512;

    unsigned long const blockOffset = gridDim.x*512;

    while ( x < lastX )
    {
        double R = (*x).x;
        double I = (*x).y;

        res += R * R + I * I;

        x += blockOffset;

    }

    if (threadIdx.x >= 32)
        sSum[threadIdx.x] = res;

    __syncthreads();

    if (threadIdx.x < 32)
        for ( unsigned long i=1; i < 16; ++i )
            res += sSum[i*32 + threadIdx.x];

    __syncthreads();

    if (threadIdx.x < 32)
    {
        double* vsSum = sSum;

        vsSum[threadIdx.x] = res;

        if (threadIdx.x < 16) vsSum[threadIdx.x] += vsSum[threadIdx.x + 16];
        __syncthreads();

        if (threadIdx.x < 8) vsSum[threadIdx.x] += vsSum[threadIdx.x + 8];
        __syncthreads();

        if (threadIdx.x < 4) vsSum[threadIdx.x] += vsSum[threadIdx.x + 4];
        __syncthreads();

        if (threadIdx.x < 2) vsSum[threadIdx.x] += vsSum[threadIdx.x + 2];
        __syncthreads();

        if (threadIdx.x == 0)
            *the_norm = sqrt( vsSum[0] + vsSum[1] );
    }
}
#endif

    //should call with Zscale<<<1, 128>>>(...);
    __global__ void Zscal( unsigned long m, double real, double2* dA )
    {
        const int i = threadIdx.x;

        for( unsigned long j = i; j < m; j += 128 ) 
        {
            dA[j].x *= real;
            dA[j].y *= real;
        }
    }

    __global__ //<<<((dim+15)/16,(dim+15)/16), (16,16)>>>
    void Zgemm( double2* P, double2* M, double2* N, unsigned long dim, double alpha )
    {
        typedef double              value_type;
        typedef double2             complex_type;
        typedef unsigned long       size_type;

        __shared__ value_type _M[16][17];
        __shared__ value_type _m[16][17];
        __shared__ value_type _N[16][17];
        __shared__ value_type _n[16][17];

        const size_type bx = blockIdx.x;
        const size_type by = blockIdx.y;
        const size_type tx = threadIdx.x;
        const size_type ty = threadIdx.y;
        const size_type row = by * 16 + ty;
        const size_type col = bx * 16 + tx;
        const size_type iter_n = (dim+15)/16;

        value_type R = 0.0;
        value_type I = 0.0;

        for ( size_type i = 0; i != iter_n; ++i )
        {
            if ( i * 16 + tx < dim && row < dim )
            {
                _M[ty][tx] = (*( M + row * dim + i * 16 + tx )).x;
                _m[ty][tx] = (*( M + row * dim + i * 16 + tx )).y;
            }
            else
            {
                _M[ty][tx] = 0.0;
                _m[ty][tx] = 0.0;
            }

            if ( i * 16 + ty < dim && col < dim )
            {
                _N[ty][tx] = (*( N + ( i * 16 + ty ) * dim + col )).x;
                _n[ty][tx] = (*( N + ( i * 16 + ty ) * dim + col )).y;
            }
            else
            {
                _N[ty][tx] = 0.0;
                _n[ty][tx] = 0.0;
            }

            __syncthreads();

            #pragma unroll
            for ( size_type j = 0; j != 16; ++j )
            {
                R += _M[ty][j] * _N[j][tx] - _m[ty][j] * _n[j][tx];
                I += _M[ty][j] * _n[j][tx] + _m[ty][j] * _N[j][tx];
            }
            __syncthreads();
        }

        if ( row < dim && col < dim )
        {
            (*( P + row * dim + col )).x = alpha * R;
            (*( P + row * dim + col )).y = alpha * I;
        }
    }

    __global__ void //<<<1,128>>>
    Zcopy( unsigned long dims, double2* src, double2* dst )
    {
        unsigned long const i = threadIdx.x;

        for( unsigned long j = i; j < dims; j += 128 ) 
        {
            (*(dst+j)).x = (*(src+j)).x;
            (*(dst+j)).y = (*(src+j)).y;
        }
    }
    __global__ void//<<<1, 128>>>
    Zaxpy( unsigned long dims, double real, double imag, double2* dst, double2* src ) // dst += (real,imag) * src
    {
        unsigned long const i = threadIdx.x;
        double R = 0.0;
        double I = 0.0;

        for( unsigned long j = i; j < dims; j += 128 ) 
        {
            R = (*(src+j)).x;
            I = (*(src+j)).y;

            (*(dst+j)).x += real * R - imag * I;
            (*(dst+j)).y += real * I + imag * R;
        }
    }

__global__ void
compose_a( double* ug, unsigned long* ar, double* diag, double thickness, double2* a, unsigned long dim )
{
    int const row_index = threadIdx.x;

    for ( unsigned long col_index = 0; col_index != dim; ++col_index )
    {
        unsigned long a_offset = row_index * dim + col_index;
        unsigned long const ug_index = *(ar+a_offset);
        //*(a+a_offset) = make_cuDoubleComplex( *(ug+ug_index+ug_index), *(ug+ug_index+ug_index+1) );
        *(a+a_offset) = make_cuDoubleComplex( -thickness * (*(ug+ug_index+ug_index+1)), thickness *( *(ug+ug_index+ug_index)) );
    }

    //*(a+row_index*dim+row_index) = make_cuDoubleComplex( *(diag+row_index), 0.0 );
    *(a+row_index*dim+row_index) = make_cuDoubleComplex( 0.0, thickness *( *(diag+row_index) ) );
}

__global__ void
extract_intensity_diff( double2* s, double* I_exp, double* I_diff, unsigned long dim, unsigned long column_index )
{
    int const I_offset = threadIdx.x;
    int const S_offset = column_index + threadIdx.x * dim;
    double const norm = cuCabs(*(s+S_offset));
    *(I_diff+I_offset) = *(I_exp+I_offset) - norm * norm;
}

__global__ void
extract_intensity_diff_with_offset( double2* s, double* I_exp, double* I_diff, unsigned long dim, unsigned long column_index, double ac_offset, double dc_offset )
{
    int const I_offset = threadIdx.x;
    int const S_offset = column_index + threadIdx.x * dim;
    double const norm = cuCabs(*(s+S_offset));
    *(I_diff+I_offset) = *(I_exp+I_offset) - norm * norm * ac_offset - dc_offset;
}


__global__ void
sum_diag( double2* a, unsigned long dim, double real, double imag )
{
    int const index = threadIdx.x;
    int const offset = index * dim + index;
    *(a+offset) = make_cuDoubleComplex( cuCreal(*(a+offset))+real, cuCimag(*(a+offset))+imag );
}

/*
 * Input/Output:
 *
 ** ug[M]
 *  ar[n][n]
 *  diag[n]         ==>>    I_diff[n]
 ** thickness
 *  dim -- n
 *  I_exp[n]
 ** column_index
 *
 *  cache:
 *  a_[n][n]    -- p2p3
 *  a^2_[n][n]  -- s
 *  a^3_[n][n]  -- s_
 *  P1[n][n]
 *  P2[n][n]
 *  P3[n][n]
 *
 * 1) compose A
 * 2) scale to A_
 * 3) compute A_^2 A_^3
 * 4) compute (P1) (P2) (P3)
 * 5) square back
 * 6) extract one column
 */
__global__ void
make_individual_pattern_intensity_diff( double* cuda_ug, unsigned long* cuda_ar, double* cuda_diag, double thickness, unsigned long* cuda_dim, double* cuda_I_exp, double* cuda_I_diff, unsigned long column_index, double2* cuda_cache, unsigned long max_dim, unsigned long tilt_size )
{
    unsigned long const tilt_index = blockDim.x * blockIdx.x + threadIdx.x;

    if ( tilt_index >= tilt_size ) return;

    unsigned long const dim = *(cuda_dim + tilt_index);
    double* ug = cuda_ug;
    unsigned long* ar = cuda_ar + tilt_index * max_dim * max_dim;
    double* diag = cuda_diag + tilt_index * max_dim;
    double* I_exp = cuda_I_exp + tilt_index * max_dim;
    double* I_diff = cuda_I_diff + tilt_index * max_dim;
    double2* cache = cuda_cache + 6 * tilt_index * max_dim * max_dim;

    unsigned long dimdim = dim*dim;

    //cache should be of size 6*N^2
    double2* a_ = cache;
    double2* aa_ = a_ + dimdim;
    double2* aaa_ = aa_ + dimdim;
    double2* p1 = aaa_ + dimdim;
    double2* p2 = p1 + dimdim;
    double2* p3 = p2 + dimdim;

    //reuse memory in latter steps, when a_, aa_ and aaa_ are idle
    //double2* p2p3 = a_;
    double2* p2p3 = aaa_;
    double2* s = aa_;
    double2* s_ = aaa_;

    //1)
    kernel_assert( (compose_a<<<1, dim>>>( ug, ar, diag, thickness, a_, dim )) );
    cuda_assert( cudaDeviceSynchronize() );

    //2)
    //TODO
    double* the_norm = (double*)aa_;
    kernel_assert( (Dznrm2<<<1,128>>>( dimdim, a_, the_norm )) );
    //kernel_assert( (Dasum<<<1,128>>>( dimdim, a_, the_norm )) );
    cuda_assert( cudaDeviceSynchronize() );
    
    //double const ratio = (*the_norm) * 53.71920351148152;
    double const ratio = (*the_norm) / 5.371920351148152;
    unsigned long const scaler = ratio < 1.0 ? 0 : ceil(log2(ratio));
    unsigned long const scaling_factor =  1 << scaler;
    double const scale = scaling_factor;
    kernel_assert( (Zscal<<<1, 128>>>( dimdim, 1.0/scale, a_ )) );    //a_ /= scale
    cuda_assert( cudaDeviceSynchronize() );

    //3)
    dim3 const mm_grids( (dim+15)/16, (dim+15)/16 );
    dim3 const mm_threads( 16, 16 );
    kernel_assert( (Zgemm<<<mm_grids, mm_threads>>>( aa_, a_, a_, dim, 1.0 )) );
    cuda_assert( cudaDeviceSynchronize() );
    kernel_assert( (Zgemm<<<mm_grids, mm_threads>>>( aaa_, aa_, a_, dim, 1.0 )) );
    cuda_assert( cudaDeviceSynchronize() );

    //4)
    /*
     * Maple:
     *  Digits := 25
     *  evalf(solve(_Z^9+9*_Z^8+72*_Z^7+504*_Z^6+3024*_Z^5+15120*_Z^4+60480*_Z^3+181440*_Z^2+362880*_Z+362880 = 0))
     * Returns:
     *  2.697333461536989227389605+5.184162062649414177834087*I,     //c1
     *  -.3810698456631129990312942+4.384644533145397950369203*I,    //c2
     *  -2.110839800302654737498705+3.089910928725500922777702*I,    //c3
     *  -3.038648072936697089212469+1.586801195758838328803868*I,    //c4
     *  -3.333551485269048803294274,                                 //c5
     *  -3.038648072936697089212469-1.586801195758838328803868*I,    //c6
     *  -2.110839800302654737498705-3.089910928725500922777702*I,    //c7
     *  -.3810698456631129990312942-4.384644533145397950369203*I,    //c8
     *  2.697333461536989227389605-5.184162062649414177834087*I      //c9
     *
     *  expand((x-c1)*(x-c2)*(x-c3))  >> p1                                                                                                   (                     p1_c                             )
     *      x^3-.205423815571221490859606*x^2-(12.65871752452031305098099*I)*x^2-58.21460179641193947200471*x-(3.189848964212376356715960*I)*x-19.71085376106750328141397+94.20645646169128946503649*I
     *
     *  expand((x-c4)*(x-c5)*(x-c6))  >> p2   (         p2_c            )
     *      x^3+9.410847631142442981719212*x^2+39.17363072664900708597702-6.123261017392618755198919*10^(-24)*I+32.01029973951970099352671*x+(4.*10^(-24)*I)*x
     *
     *  expand((x-c7)*(x-c8)*(x-c9))  >> p3                                                                                                  (                         p3_c                         )
     *      x^3-.205423815571221490859601*x^2+(12.65871752452031305098099*I)*x^2-58.21460179641193947200470*x+(3.18984896421237635671600*I)*x-19.71085376106750328141404-94.20645646169128946503646*I
     *
     *  expand((x-c1)*(x-c2)*(x-c3)*(x-c4)*(x-c5)*(x-c6)*(x-c7)*(x-c8)*(x-c9))
     *      3.628800000000000000000003*10^5-1.365022562699469279472268*10^(-19)*I+3.628800000000000000000003*10^5*x+x^9+9.00000000000000000000000*x^8+72.00000000000000000000006*x^7+503.9999999999999999999995*x^6+3024.000000000000000000002*x^5+15120.00000000000000000000*x^4+60479.99999999999999999995*x^3+1.814400000000000000000001*10^5*x^2-(5.*10^(-22)*I)*x^6-(1.*10^(-20)*I)*x^4-(1.0*10^(-19)*I)*x^3+(2.*10^(-24)*I)*x^8-(3.0*10^(-19)*I)*x^2-(7.*10^(-21)*I)*x^5-(4.*10^(-19)*I)*x+(2.*10^(-23)*I)*x^7
     */
    //4 - p1)
    kernel_assert( (Zcopy<<<1,128>>>( dimdim, aaa_, p1 )) );
    cuda_assert( cudaDeviceSynchronize() );
    kernel_assert( (Zaxpy<<<1,128>>>( dimdim, -0.205423815571221490859606, -12.65871752452031305098099, p1, aa_ )) );
    cuda_assert( cudaDeviceSynchronize() );
    kernel_assert( (Zaxpy<<<1,128>>>( dimdim, -58.21460179641193947200471, -3.189848964212376356715960, p1, a_ )) );
    cuda_assert( cudaDeviceSynchronize() );
    kernel_assert( (sum_diag<<<1,dim>>>( p1, dim, -19.71085376106750328141397, 94.20645646169128946503649 )) );
    cuda_assert( cudaDeviceSynchronize() );

    //4 - p2)
    kernel_assert( (Zcopy<<<1,128>>>( dimdim, aaa_, p2 )) );
    cuda_assert( cudaDeviceSynchronize() );
    kernel_assert( (Zaxpy<<<1,128>>>( dimdim, 9.410847631142442981719212, 0.0, p2, aa_ )) );
    cuda_assert( cudaDeviceSynchronize() );
    kernel_assert( (Zaxpy<<<1,128>>>( dimdim, 32.01029973951970099352671, 0.0, p2, a_ )) );
    cuda_assert( cudaDeviceSynchronize() );
    kernel_assert( (sum_diag<<<1,dim>>>( p2, dim, 39.17363072664900708597702, 0.0  )) );
    cuda_assert( cudaDeviceSynchronize() );

    //4 - p3)
    kernel_assert( (Zcopy<<<1,128>>>( dimdim, aaa_, p3 )) );
    cuda_assert( cudaDeviceSynchronize() );
    kernel_assert( (Zaxpy<<<1,128>>>( dimdim, -0.205423815571221490859601, 12.65871752452031305098099, p3, aa_ )) );
    cuda_assert( cudaDeviceSynchronize() );
    kernel_assert( (Zaxpy<<<1,128>>>( dimdim, -58.21460179641193947200470, 3.18984896421237635671600, p3, a_ )) );
    cuda_assert( cudaDeviceSynchronize() );
    kernel_assert( (sum_diag<<<1,dim>>>( p3, dim, -19.71085376106750328141404, -94.20645646169128946503646 )) );
    cuda_assert( cudaDeviceSynchronize() );

    //4 - s)
    // s = 1/602.39521910453439454428( p1 * ( 1/602.39521910453439454428 * p2 * p3 ) ) = (p1 p2 p3)/362880
    kernel_assert( (Zgemm<<<mm_grids, mm_threads>>>( p2p3, p2, p3, dim, 0.0016600397351866578333 )) );
    cuda_assert( cudaDeviceSynchronize() );
    kernel_assert( (Zgemm<<<mm_grids, mm_threads>>>( s, p1, p2p3, dim, 0.0016600397351866578333 )) );
    cuda_assert( cudaDeviceSynchronize() );

    //5)
    if ( scaler != 0 )
    {
        for ( unsigned long index = 0; index != scaler; ++index )
        {
            kernel_assert( (Zgemm<<<mm_grids, mm_threads>>>( s_, s, s, dim, 1.0 )) );
            cuda_assert( cudaDeviceSynchronize() );
            double2* tmp = s_;
            s_ = s;
            s = tmp;
        }
    }

    //6)
    //kernel_assert( (extract_intensity_diff<<<1,dim>>>( s, I_exp, I_diff, dim, column_index )) );
    double const ac_offset = cuda_ug[0];
    double const dc_offset = cuda_ug[1];
    kernel_assert( (extract_intensity_diff_with_offset<<<1,dim>>>( s, I_exp, I_diff, dim, column_index, ac_offset, dc_offset )) );
    cuda_assert( cudaDeviceSynchronize() );
}

void make_pattern_intensity_diff( double* cuda_ug, unsigned long* cuda_ar, double* cuda_diag, double thickness, unsigned long* cuda_dim, double* cuda_I_exp, double* cuda_I_diff, unsigned long column_index, double2* cuda_cache, unsigned long tilt_size, unsigned long max_dim )
{
    unsigned long const threads = 64;
    unsigned long const grids = (tilt_size + threads - 1)/threads;

    kernel_assert( ( make_individual_pattern_intensity_diff<<<grids, threads>>>( cuda_ug, cuda_ar, cuda_diag, thickness, cuda_dim, cuda_I_exp, cuda_I_diff, column_index, cuda_cache, max_dim, tilt_size ) ) );
    cuda_assert( cudaDeviceSynchronize() );
}

