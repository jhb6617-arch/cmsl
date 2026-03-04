#include "header.h"
#include "kernels_evol.h"

__global__ void kernelAddSmallNoise_Ps(double* rand_d, double* Ps_old_d,
                                       double noise_level,
                                       int nx, int ny, int nz)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int iz = threadIdx.z + blockIdx.z * blockDim.z;
    if (ix >= (unsigned)nx || iy >= (unsigned)ny || iz >= (unsigned)nz)
        return;

    const unsigned int id = ix + nx * (iy + iz * ny);

    // rand_d[id] is in [0,1), make it symmetric around 0: (-0.5 to 0.5)
    double r = rand_d[id] - 0.5;

    // Add tiny dimensionless noise
    Ps_old_d[id] += noise_level * r;
}

__global__ void build_mat_mask_and_p(
    const double* __restrict__ Pmz,
    const int*    __restrict__ omega,
    const int*    __restrict__ grain_mat,
    double* __restrict__ Pmz_mat,
    double* __restrict__ mask_mat,
    int Nvox, int target_mat)   // 0=AFE,1=FE,2=DE
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= Nvox) return;

    int gid = omega[id];   // voxel -> grain id (or -1)
    double m = 0.0;

    if (gid >= 0) {
        int mat = grain_mat[gid];
        if (mat == target_mat) m = 1.0;
    }

    mask_mat[id] = m;
    Pmz_mat[id]  = m * Pmz[id];
}


__global__ void kernelCalc_antiferro(
    double* Pmx_d, double* Pmx_old_d, double* Psx_d, double* Psx_old_d,
    double* Pmy_d, double* Pmy_old_d, double* Psy_d, double* Psy_old_d,
    double* Pmz_d, double* Pmz_old_d, double* Psz_d, double* Psz_old_d,
    int* omega_d, const double* TR_d, const double* alpha_m, 
    const double* beta_m,  const double* gamma_m, const double* g_m,     const double* a0_m,
    const double* kappa_nd_m, const double* L_nd_m, double del_h, double del_t, 
    double *delecX_d, double *delecY_d, double *delecZ_d,
    int dep_flag,    int nx, int ny, int nz, int ng_total,
    int count, double Ez_ext, const double g_top, const double g_bot,  double *Pmx_local_d, 
    double *Psx_local_d, double *Pmy_local_d, double *Psy_local_d,   double* Pmz_local_d,  
    double* Psz_local_d, const int* __restrict__ is_interface_d)
{
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= nx || j >= ny || k >= nz) return;

    const unsigned int id = i + nx * (j + k * ny);

    const int gid = omega_d[id];
    if (gid < 0 || gid >= ng_total) return;   // safety guard

    // neighbor indices (x,y periodic; z clamped)
    const int e = (i + 1) % nx;
    const int w = (i - 1 + nx) % nx;
    const int n = (j + 1) % ny;
    const int s = (j - 1 + ny) % ny;
    const int u = (k == nz - 1) ? k : (k + 1);
    const int d = (k == 0)      ? k : (k - 1);

    const int idx_e = e + nx * (j + k * ny);
    const int idx_w = w + nx * (j + k * ny);
    const int idx_n = i + nx * (n + k * ny);
    const int idx_s = i + nx * (s + k * ny);
    const int idx_u = i + nx * (j + u * ny);
    const int idx_d = i + nx * (j + d * ny);

    // --- All P* are already nondimensional ---
    // M neighbors
    double pmx1 = Pmx_old_d[idx_e], pmy1 = Pmy_old_d[idx_e], pmz1 = Pmz_old_d[idx_e];
    double pmx2 = Pmx_old_d[idx_w], pmy2 = Pmy_old_d[idx_w], pmz2 = Pmz_old_d[idx_w];
    double pmx3 = Pmx_old_d[idx_n], pmy3 = Pmy_old_d[idx_n], pmz3 = Pmz_old_d[idx_n];
    double pmx4 = Pmx_old_d[idx_s], pmy4 = Pmy_old_d[idx_s], pmz4 = Pmz_old_d[idx_s];
    double pmx5 = Pmx_old_d[idx_u], pmy5 = Pmy_old_d[idx_u], pmz5 = Pmz_old_d[idx_u];
    double pmx6 = Pmx_old_d[idx_d], pmy6 = Pmy_old_d[idx_d], pmz6 = Pmz_old_d[idx_d];

    //double pmx5 = (k == nz - 1) ? g_top : Pmx_old_d[idx_u];
    //double pmy5 = (k == nz - 1) ? g_top : Pmy_old_d[idx_u];
    //double pmz5 = (k == nz - 1) ? g_top : Pmz_old_d[idx_u];
    //double pmx6 = (k == 0) ? g_bot : Pmx_old_d[idx_d];
    //double pmy6 = (k == 0) ? g_bot : Pmy_old_d[idx_d];
    //double pmz6 = (k == 0) ? g_bot : Pmz_old_d[idx_d];

    const double phiMxsum = pmx1 + pmx2 + pmx3 + pmx4 + pmx5 + pmx6;
    const double phiMysum = pmy1 + pmy2 + pmy3 + pmy4 + pmy5 + pmy6;
    const double phiMzsum = pmz1 + pmz2 + pmz3 + pmz4 + pmz5 + pmz6;

    // S neighbors
    double psx1 = Psx_old_d[idx_e], psy1 = Psy_old_d[idx_e], psz1 = Psz_old_d[idx_e];
    double psx2 = Psx_old_d[idx_w], psy2 = Psy_old_d[idx_w], psz2 = Psz_old_d[idx_w];
    double psx3 = Psx_old_d[idx_n], psy3 = Psy_old_d[idx_n], psz3 = Psz_old_d[idx_n];
    double psx4 = Psx_old_d[idx_s], psy4 = Psy_old_d[idx_s], psz4 = Psz_old_d[idx_s];
    double psx5 = Psx_old_d[idx_u], psy5 = Psy_old_d[idx_u], psz5 = Psz_old_d[idx_u];
    double psx6 = Psx_old_d[idx_d], psy6 = Psy_old_d[idx_d], psz6 = Psz_old_d[idx_d];

    //double psx5 = (k == nz - 1) ? g_top : Psx_old_d[idx_u];
    //double psy5 = (k == nz - 1) ? g_top : Psy_old_d[idx_u];
    //double psz5 = (k == nz - 1) ? g_top : Psz_old_d[idx_u];
    //double psx6 = (k == 0) ? g_bot : Psx_old_d[idx_d];
    //double psy6 = (k == 0) ? g_bot : Psy_old_d[idx_d];
    //double psz6 = (k == 0) ? g_bot : Psz_old_d[idx_d];

    const double phiSxsum = psx1 + psx2 + psx3 + psx4 + psx5 + psx6;
    const double phiSysum = psy1 + psy2 + psy3 + psy4 + psy5 + psy6;
    const double phiSzsum = psz1 + psz2 + psz3 + psz4 + psz5 + psz6;

    // Center values (nondim)
    const double pMx0 = Pmx_old_d[id];
    const double pMy0 = Pmy_old_d[id];
    const double pMz0 = Pmz_old_d[id];
    const double pSx0 = Psx_old_d[id];
    const double pSy0 = Psy_old_d[id];
    const double pSz0 = Psz_old_d[id];

    // --- load per-voxel parameters ---
    const double alpha_nd = alpha_m[id];  
    const double beta_nd  = beta_m[id];
    const double gamma_nd = gamma_m[id];
    double g_loc     = g_m[id];
    const double a0_nd    = a0_m[id];

    const double kappa_nd = kappa_nd_m[id];
    double Lfac     = L_nd_m[id];
    // ---- interface stabilization (FE-only voxels are flagged) ----
    if (is_interface_d[id]) {
       // FE stabilization near interfaces:
       g_loc *= 0.4;
       Lfac *= (1.0 + 10.0);
    }

    const double alpha_g_nd   = 2.0 * (alpha_nd + g_loc);
    const double alpha_g_nd_s = 2.0 * (alpha_nd - g_loc);
    const double beta_4  = 4.0  * beta_nd;
    const double beta_12 = 12.0 * beta_nd;
    const double gamma_6 = 6.0  * gamma_nd;
    const double a0_2    = 2.0  * a0_nd;


    // Orientation transform
    const double tr11 = TR_d[gid  * 9 + 0], tr12 = TR_d[gid  * 9 + 1], tr13 = TR_d[gid  * 9 + 2];
    const double tr21 = TR_d[gid  * 9 + 3], tr22 = TR_d[gid  * 9 + 4], tr23 = TR_d[gid  * 9 + 5];
    const double tr31 = TR_d[gid  * 9 + 6], tr32 = TR_d[gid  * 9 + 7], tr33 = TR_d[gid  * 9 + 8];

    // Rotate P to crystal frame (still nondim)
    const double pMXl = tr11 * pMx0 + tr12 * pMy0 + tr13 * pMz0;
    const double pMYl = tr21 * pMx0 + tr22 * pMy0 + tr23 * pMz0;
    const double pMZl = tr31 * pMx0 + tr32 * pMy0 + tr33 * pMz0;

    const double pSXl = tr11 * pSx0 + tr12 * pSy0 + tr13 * pSz0;
    const double pSYl = tr21 * pSx0 + tr22 * pSy0 + tr23 * pSz0;
    const double pSZl = tr31 * pSx0 + tr32 * pSy0 + tr33 * pSz0;

    Pmx_local_d[id] = pMXl;
    Psx_local_d[id] = pSXl;
    Pmy_local_d[id] = pMYl;
    Psy_local_d[id] = pSYl;
    Pmz_local_d[id] = pMZl;
    Psz_local_d[id] = pSZl;

    // Landau derivatives
    double dfdPmxl = a0_2 * pMXl;
    double dfdPmyl = a0_2 * pMYl;
    double dfdPsxl = a0_2 * pSXl;
    double dfdPsyl = a0_2 * pSYl;

    const double phiMz2 = pMZl * pMZl;
    const double phiMz3 = phiMz2 * pMZl;
    const double phiMz4 = phiMz3 * pMZl;
    const double phiMz5 = phiMz4 * pMZl;

    const double phiSz2 = pSZl * pSZl;
    const double phiSz3 = phiSz2 * pSZl;
    const double phiSz4 = phiSz3 * pSZl;
    const double phiSz5 = phiSz4 * pSZl;

    double dfdPmzl = alpha_g_nd * pMZl
                   + beta_4  * phiMz3 + beta_12 * pMZl * phiSz2
                   + gamma_6 * (phiMz5 + 10.0 * phiMz3 * phiSz2 + 5.0 * pMZl * phiSz4);

    double dfdPszl = alpha_g_nd_s * pSZl
                   + beta_4  * phiSz3 + beta_12 * pSZl * phiMz2
                   + gamma_6 * (phiSz5 + 10.0 * phiMz2 * phiSz3 + 5.0 * pSZl * phiMz4);

    // rotate derivatives back to lab frame
    double dfdPmx_b = tr11 * dfdPmxl + tr21 * dfdPmyl + tr31 * dfdPmzl;
    double dfdPmy_b = tr12 * dfdPmxl + tr22 * dfdPmyl + tr32 * dfdPmzl;
    double dfdPmz_b = tr13 * dfdPmxl + tr23 * dfdPmyl + tr33 * dfdPmzl;

    double dfdPsx_b = tr11 * dfdPsxl + tr21 * dfdPsyl + tr31 * dfdPszl;
    double dfdPsy_b = tr12 * dfdPsxl + tr22 * dfdPsyl + tr32 * dfdPszl;
    double dfdPsz_b = tr13 * dfdPsxl + tr23 * dfdPsyl + tr33 * dfdPszl;

    const double dfdPmz_app = - Ez_ext;

    const double A_loc    = kappa_nd * del_t / (del_h * del_h);

    double dfdPmx_e =  (dep_flag != 0) ? - 1.0 * delecX_d[id] : 0.0;
    double dfdPmy_e =  (dep_flag != 0) ? - 1.0 * delecY_d[id] : 0.0;
    double dfdPmz_e =  (dep_flag != 0) ? - 1.0 * delecZ_d[id] : 0.0;

    dfdPmx_b += dfdPmx_e;
    dfdPmy_b += dfdPmy_e;
    dfdPmz_b += dfdPmz_app + dfdPmz_e;

    // Update (explicit Euler with 6-nb Laplacian)
    Pmx_d[id] = pMx0 * (1.0 - 6.0 * A_loc) + A_loc * phiMxsum - Lfac * del_t * dfdPmx_b;
    Pmy_d[id] = pMy0 * (1.0 - 6.0 * A_loc) + A_loc * phiMysum - Lfac * del_t * dfdPmy_b;
    Pmz_d[id] = pMz0 * (1.0 - 6.0 * A_loc) + A_loc * phiMzsum - Lfac * del_t * dfdPmz_b;

    Psx_d[id] = pSx0 * (1.0 - 6.0 * A_loc) + A_loc * phiSxsum - Lfac * del_t * dfdPsx_b;
    Psy_d[id] = pSy0 * (1.0 - 6.0 * A_loc) + A_loc * phiSysum - Lfac * del_t * dfdPsy_b;
    Psz_d[id] = pSz0 * (1.0 - 6.0 * A_loc) + A_loc * phiSzsum - Lfac * del_t * dfdPsz_b;

}

__global__ void kernelUpdate_P(double *x_d, double *x_old_d, double *y_d, double *y_old_d,  double* z_d, double* z_old_d, int nx, int ny, int nz){
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

  if ( id < nx * ny * nz) {
    x_old_d[id] = x_d[id];
    y_old_d[id] = y_d[id];
    z_old_d[id] = z_d[id];
  }

}
