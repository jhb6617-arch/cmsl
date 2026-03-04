#ifndef CUDA_DEPS_H
#define CUDA_DEPS_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cusparse.h>

extern dim3 threads, blocks;
extern int threadsPerBlock, numBlocks;

extern double *Pmx_old_d, *Pmy_old_d, *Pmz_old_d;
extern double *Pmx_d, *Pmy_d, *Pmz_d;

extern double *Psx_old_d, *Psy_old_d, *Psz_old_d;
extern double *Psx_d, *Psy_d, *Psz_d;

extern double *phi_old_d;
extern double *phi_dep_d;
extern bool phi_dep_initialized;

// extra device pointers
extern double *delecX_d, *delecY_d, *delecZ_d;
extern double *divP_d;

extern double *eta_d;
extern int    *omega_d;

extern double *phi_d, *theta_d, *psi_d;
extern double *TR_d;

extern double *Pmx_local_d, *Pmy_local_d, *Pmz_local_d;
extern double *Psx_local_d, *Psy_local_d, *Psz_local_d;

extern cudaEvent_t start, stop;
extern double *Psum_d;
extern void   *t_storage;
extern size_t  t_storage_bytes;

extern int *d_grain_mat;

#ifdef __CUDACC__
extern __device__ __constant__ int d_zbc_type;
extern __device__ int d_ref_id;
#endif

#endif
