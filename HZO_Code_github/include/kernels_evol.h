#ifndef KERNELS_H
#define KERNELS_H

__global__ void kernelAddSmallNoise_Ps(double* rand_d, double* Ps_old_d,
                                       double noise_level,
                                       int nx, int ny, int nz);


__global__ void kernelCalc_antiferro(
    double* Pmx_d, double* Pmx_old_d, double* Psx_d, double* Psx_old_d,
    double* Pmy_d, double* Pmy_old_d, double* Psy_d, double* Psy_old_d,
    double* Pmz_d, double* Pmz_old_d, double* Psz_d, double* Psz_old_d,
    int* omega_d, const double* TR_d,         const double* alpha_m, const double* beta_m,  const double* gamma_m,
    const double* g_m,     const double* a0_m, const double* kappa_nd_m, const double* L_nd_m, 
    double del_h, double del_t, double *delecX_d, double *delecY_d, double *delecZ_d,
    int dep_flag, int nx, int ny, int nz, int ng, int count, double Ez_ext, const double g_top, const double g_bot,
    double *Pmx_local_d, double *Psx_local_d, double *Pmy_local_d, double *Psy_local_d,    
    double* Pmz_local_d,   double* Psz_local_d, const int* __restrict__ is_interface_d);

__global__ void kernelUpdate_P(double *x_d, double *x_old_d, double *y_d, 
		double *y_old_d,  double* z_d, double* z_old_d, int nx, int ny, int nz);

__global__ void build_fe_mask_and_p(
    const double* __restrict__ Pmz,
    const int*    __restrict__ omega,
    const int*    __restrict__ grain_mat,
    double* __restrict__ Pmz_FE,
    double* __restrict__ mask_FE,
    int Nvox);

__global__ void build_mat_mask_and_p(
    const double* __restrict__ Pmz,
    const int*    __restrict__ omega,
    const int*    __restrict__ grain_mat,
    double* __restrict__ Pmz_mat,
    double* __restrict__ mask_mat,
    int Nvox,
    int target_mat);

#endif
