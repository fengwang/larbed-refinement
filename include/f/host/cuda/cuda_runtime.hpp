#ifndef KMKQOEUFEQJINBNUNIOOGWVSDAVGPURLLECEHBYBBWKDUEMECTKYKBSXBFNKVSGDOLWKFIIIF
#define KMKQOEUFEQJINBNUNIOOGWVSDAVGPURLLECEHBYBBWKDUEMECTKYKBSXBFNKVSGDOLWKFIIIF

extern "C"
{

    typedef enum cudaError
    {
        cudaSuccess = 0,
        cudaErrorMissingConfiguration = 1,
        cudaErrorMemoryAllocation = 2,
        cudaErrorInitializationError = 3,
        cudaErrorLaunchFailure = 4,
        cudaErrorPriorLaunchFailure = 5,
        cudaErrorLaunchTimeout = 6,
        cudaErrorLaunchOutOfResources = 7,
        cudaErrorInvalidDeviceFunction = 8,
        cudaErrorInvalidConfiguration = 9,
        cudaErrorInvalidDevice = 10,
        cudaErrorInvalidValue = 11,
        cudaErrorInvalidPitchValue = 12,
        cudaErrorInvalidSymbol = 13,
        cudaErrorMapBufferObjectFailed = 14,
        cudaErrorUnmapBufferObjectFailed = 15,
        cudaErrorInvalidHostPointer = 16,
        cudaErrorInvalidDevicePointer = 17,
        cudaErrorInvalidTexture = 18,
        cudaErrorInvalidTextureBinding = 19,
        cudaErrorInvalidChannelDescriptor = 20,
        cudaErrorInvalidMemcpyDirection = 21,
        cudaErrorAddressOfConstant = 22,
        cudaErrorTextureFetchFailed = 23,
        cudaErrorTextureNotBound = 24,
        cudaErrorSynchronizationError = 25,
        cudaErrorInvalidFilterSetting = 26,
        cudaErrorInvalidNormSetting = 27,
        cudaErrorMixedDeviceExecution = 28,
        cudaErrorCudartUnloading = 29,
        cudaErrorUnknown = 30,
        cudaErrorNotYetImplemented = 31,
        cudaErrorMemoryValueTooLarge = 32,
        cudaErrorInvalidResourceHandle = 33,
        cudaErrorNotReady = 34,
        cudaErrorInsufficientDriver = 35,
        cudaErrorSetOnActiveProcess = 36,
        cudaErrorInvalidSurface = 37,
        cudaErrorNoDevice = 38,
        cudaErrorECCUncorrectable = 39,
        cudaErrorSharedObjectSymbolNotFound = 40,
        cudaErrorSharedObjectInitFailed = 41,
        cudaErrorUnsupportedLimit = 42,
        cudaErrorDuplicateVariableName = 43,
        cudaErrorDuplicateTextureName = 44,
        cudaErrorDuplicateSurfaceName = 45,
        cudaErrorDevicesUnavailable = 46,
        cudaErrorInvalidKernelImage = 47,
        cudaErrorNoKernelImageForDevice = 48,
        cudaErrorIncompatibleDriverContext = 49,
        cudaErrorPeerAccessAlreadyEnabled = 50,
        cudaErrorPeerAccessNotEnabled = 51,
        cudaErrorDeviceAlreadyInUse = 54,
        cudaErrorProfilerDisabled = 55,
        cudaErrorProfilerNotInitialized = 56,
        cudaErrorProfilerAlreadyStarted = 57,
        cudaErrorProfilerAlreadyStopped = 58,
        cudaErrorAssert = 59,
        cudaErrorTooManyPeers = 60,
        cudaErrorHostMemoryAlreadyRegistered = 61,
        cudaErrorHostMemoryNotRegistered = 62,
        cudaErrorOperatingSystem = 63,
        cudaErrorPeerAccessUnsupported = 64,
        cudaErrorLaunchMaxDepthExceeded = 65,
        cudaErrorLaunchFileScopedTex = 66,
        cudaErrorLaunchFileScopedSurf = 67,
        cudaErrorSyncDepthExceeded = 68,
        cudaErrorLaunchPendingCountExceeded = 69,
        cudaErrorNotPermitted = 70,
        cudaErrorNotSupported = 71,
        cudaErrorHardwareStackError = 72,
        cudaErrorIllegalInstruction = 73,
        cudaErrorMisalignedAddress = 74,
        cudaErrorInvalidAddressSpace = 75,
        cudaErrorInvalidPc = 76,
        cudaErrorIllegalAddress = 77,
        cudaErrorInvalidPtx = 78,
        cudaErrorInvalidGraphicsContext = 79,
        cudaErrorStartupFailure = 0x7f,
        cudaErrorApiFailureBase = 10000
    } cudaError_t;

    enum cudaMemcpyKind
    {
        cudaMemcpyHostToHost = 0,
        cudaMemcpyHostToDevice = 1,
        cudaMemcpyDeviceToHost = 2,
        cudaMemcpyDeviceToDevice = 3,
        cudaMemcpyDefault = 4
    };

    cudaError_t cudaMalloc( void** devPtr, unsigned long int size );
    cudaError_t cudaMemset( void* devPtr, int value, unsigned long int count );
    cudaError_t cudaFree( void* devPtr );
    cudaError_t cudaSetDevice( int device );
    cudaError_t cudaGetDevice( int *device );
    cudaError_t cudaMemcpy( void* dst, const void* src, unsigned long int count, enum cudaMemcpyKind kind );
    cudaError_t cudaDeviceSynchronize();
    cudaError_t cudaDeviceReset();
    cudaError_t cudaMallocManaged ( void** devPtr, unsigned long size, unsigned int  flags );

}

#endif//KMKQOEUFEQJINBNUNIOOGWVSDAVGPURLLECEHBYBBWKDUEMECTKYKBSXBFNKVSGDOLWKFIIIF

