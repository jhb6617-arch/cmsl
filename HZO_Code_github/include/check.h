#include <stdarg.h>
#include<stdio.h>
#include <sys/stat.h>


/***********************************************************************************
                         Check errors from CUDA API calls
 ***********************************************************************************/
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

/***********************************************************************************
                         Check errors from CUDA kernel calls
 ***********************************************************************************/
#define CHECK_KERNEL()                                                         \
{                                                                              \
    const cudaError_t error = cudaGetLastError();                              \
    if ( cudaSuccess != error )                                                \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}


/***********************************************************************************
                        Check errors from cuFFT API calls
 ***********************************************************************************/
#define CHECK_CUFFT(call)                                                      \
{                                                                              \
    const cufftResult error = call;                                            \
    if (error != CUFFT_SUCCESS)                                                \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "cuFFT error code: %d\n", error);                      \
        exit(1);                                                               \
    }                                                                          \
}

// CUDA error check
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)


// cuBLAS error check
#define CHECK_CUBLAS(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = call;                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n",                     \
                    __FILE__, __LINE__, status);                               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// cuSPARSE error check
#define CHECK_CUSPARSE(call)                                                   \
    do {                                                                       \
        cusparseStatus_t status = call;                                        \
        if (status != CUSPARSE_STATUS_SUCCESS) {                               \
            fprintf(stderr, "cuSPARSE error at %s:%d: %d\n",                   \
                    __FILE__, __LINE__, status);                               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
} while (0)

static inline void safe_snprintf(char *dst, size_t dstsz, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(dst, dstsz, fmt, ap);
    va_end(ap);

    if (n < 0 || (size_t)n >= dstsz) {
        fprintf(stderr, "Path truncated/overflow in formatting: %s\n", fmt);
        exit(1);
    }
}
static void make_dir_p(const char *path)
{
    char cmd[PATHBUF];
    safe_snprintf(cmd, sizeof(cmd), "mkdir -p -- '%s'", path);

    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "mkdir -p failed for %s (ret=%d)\n", path, ret);
        exit(1);
    }
}

static void set_output_dir_for_structure(int sid)
{
    snprintf(output_dir, sizeof(output_dir),
             "output/struc%03d", sid);
    make_dir_p(output_dir);     // mkdir -p output/strcu01
    printf("Output dir set to: %s\n", output_dir);
}

