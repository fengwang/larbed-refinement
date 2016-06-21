#ifndef YYLHOHUFOPIMNPPGVGACFIDLOSOXDBVXHUINBYULQBJWMQYVPEIQBPDBQPXBGUJCYGDQDXHYQ
#define YYLHOHUFOPIMNPPGVGACFIDLOSOXDBVXHUINBYULQBJWMQYVPEIQBPDBQPXBGUJCYGDQDXHYQ

extern "C"
{

    typedef enum
    {
        CUBLAS_STATUS_SUCCESS = 0,
        CUBLAS_STATUS_NOT_INITIALIZED = 1,
        CUBLAS_STATUS_ALLOC_FAILED = 3,
        CUBLAS_STATUS_INVALID_VALUE = 7,
        CUBLAS_STATUS_ARCH_MISMATCH = 8,
        CUBLAS_STATUS_MAPPING_ERROR = 11,
        CUBLAS_STATUS_EXECUTION_FAILED = 13,
        CUBLAS_STATUS_INTERNAL_ERROR = 14,
        CUBLAS_STATUS_NOT_SUPPORTED = 15,
        CUBLAS_STATUS_LICENSE_ERROR = 16
    } cublasStatus_t;

    struct cublasContext;
    typedef struct cublasContext* cublasHandle_t;

    cublasStatus_t cublasCreate_v2 ( cublasHandle_t* handle );
    cublasStatus_t cublasDestroy_v2 ( cublasHandle_t handle );
    cublasStatus_t cublasDnrm2_v2( cublasHandle_t handle, int n, const double* x, int incx, double* result );
    cublasStatus_t cublasDasum_v2( cublasHandle_t handle, int n, const double* x, int incx, double* result );
    cublasStatus_t cublasDdot_v2 (cublasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result);

}

#endif//YYLHOHUFOPIMNPPGVGACFIDLOSOXDBVXHUINBYULQBJWMQYVPEIQBPDBQPXBGUJCYGDQDXHYQ

