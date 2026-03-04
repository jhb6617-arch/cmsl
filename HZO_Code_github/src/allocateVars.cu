#include "header.h"
#include "check.h"
#include <stdlib.h>

void AllocateData()
{
    size_t nvox = (size_t)nx * (size_t)ny * (size_t)nz;

    // --- per-voxel Landau params (device) ---
    CHECK(cudaMalloc((void**)&alpha_m,    nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&beta_m,     nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&gamma_m,    nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&g_m,        nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&eps_m,        nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&a0_m,       nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&kappa_nd_m, nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&L_nd_m,     nvox * sizeof(double)));

    // Optional: memset to catch size mistakes early
    CHECK(cudaMemset(alpha_m,    0, nvox * sizeof(double)));
    CHECK(cudaMemset(beta_m,     0, nvox * sizeof(double)));
    CHECK(cudaMemset(gamma_m,    0, nvox * sizeof(double)));
    CHECK(cudaMemset(g_m,        0, nvox * sizeof(double)));
    CHECK(cudaMemset(eps_m,      0, nvox * sizeof(double)));
    CHECK(cudaMemset(a0_m,       0, nvox * sizeof(double)));
    CHECK(cudaMemset(kappa_nd_m, 0, nvox * sizeof(double)));
    CHECK(cudaMemset(L_nd_m,     0, nvox * sizeof(double)));

    CHECK_CUDA(cudaMalloc ((void**)&delecX_d,sizeof(double) * nvox));
    CHECK_CUDA(cudaMalloc ((void**)&delecY_d,sizeof(double) * nvox));
    CHECK_CUDA(cudaMalloc ((void**)&delecZ_d,sizeof(double) * nvox));
    CHECK_CUDA(cudaMalloc ((void**)&divP_d,sizeof(double) * nvox));

    CHECK_CUDA(cudaMemset(delecX_d, 0, nx * ny * nz *sizeof(double))); 
    CHECK_CUDA(cudaMemset(delecY_d, 0, nx * ny * nz *sizeof(double))); 
    CHECK_CUDA(cudaMemset(delecZ_d, 0, nx * ny * nz *sizeof(double)));
    CHECK_CUDA(cudaMemset(divP_d, 0, nx * ny * nz *sizeof(double)));

    // --- device arrays ---
    CHECK(cudaMalloc((void**)&eta_d,    nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&omega_d,  nvox * sizeof(int)));

    CHECK(cudaMalloc((void**)&phi_d,   sizeof(double) * ng_total));
    CHECK(cudaMalloc((void**)&theta_d, sizeof(double) * ng_total));
    CHECK(cudaMalloc((void**)&psi_d,   sizeof(double) * ng_total));
    CHECK(cudaMalloc((void**)&TR_d,    sizeof(double) * ng_total * 9));

    // --- host arrays ---
    TR_h   = (double*)malloc(sizeof(double) * ng_total * 9);
    eta    = (double*)malloc(sizeof(double) * nvox);
    omega  = (int*)   malloc(sizeof(int)    * nvox);
    phi    = (double*)malloc(sizeof(double) * ng_total);
    theta  = (double*)malloc(sizeof(double) * ng_total);
    psi    = (double*)malloc(sizeof(double) * ng_total);
    centx  = (double*)malloc(sizeof(double) * ng_total);
    centy  = (double*)malloc(sizeof(double) * ng_total);
    centz  = (double*)malloc(sizeof(double) * ng_total);

    inp_ga = (double*)malloc(sizeof(double) * n_samples);
    out_ga = (double*)malloc(sizeof(double) * n_samples);

    CHECK(cudaMalloc((void**)&Pmx_old_d, nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&Pmy_old_d, nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&Pmz_old_d, nvox * sizeof(double)));
    CHECK(cudaMemset(Pmx_old_d, 0, nvox * sizeof(double)));
    CHECK(cudaMemset(Pmy_old_d, 0, nvox * sizeof(double)));
    CHECK(cudaMemset(Pmz_old_d, 0, nvox * sizeof(double)));

    CHECK(cudaMalloc((void**)&Psx_old_d, nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&Psy_old_d, nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&Psz_old_d, nvox * sizeof(double)));
    CHECK(cudaMemset(Psx_old_d, 0, nvox * sizeof(double)));
    CHECK(cudaMemset(Psy_old_d, 0, nvox * sizeof(double)));
    CHECK(cudaMemset(Psz_old_d, 0, nvox * sizeof(double)));

    CHECK(cudaMalloc((void**)&Pmx_d, nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&Pmy_d, nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&Pmz_d, nvox * sizeof(double)));
    CHECK(cudaMemset(Pmx_d, 0, nvox * sizeof(double)));
    CHECK(cudaMemset(Pmy_d, 0, nvox * sizeof(double)));
    CHECK(cudaMemset(Pmz_d, 0, nvox * sizeof(double)));

    CHECK(cudaMalloc((void**)&Psx_d, nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&Psy_d, nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&Psz_d, nvox * sizeof(double)));
    CHECK(cudaMemset(Psx_d, 0, nvox * sizeof(double)));
    CHECK(cudaMemset(Psy_d, 0, nvox * sizeof(double)));
    CHECK(cudaMemset(Psz_d, 0, nvox * sizeof(double)));

    CHECK_CUDA(cudaMalloc(&phi_old_d, nvox * sizeof(double)));
    CHECK_CUDA(cudaMemset(phi_old_d, 0, nvox * sizeof(double)));

    CHECK(cudaMalloc((void**)&Pmx_local_d, nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&Pmy_local_d, nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&Pmz_local_d, nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&Psx_local_d, nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&Psy_local_d, nvox * sizeof(double)));
    CHECK(cudaMalloc((void**)&Psz_local_d, nvox * sizeof(double)));
    
    CHECK_CUDA(cudaMemset(Pmx_local_d, 0, nvox * sizeof(double)));
    CHECK_CUDA(cudaMemset(Pmy_local_d, 0, nvox * sizeof(double)));
    CHECK_CUDA(cudaMemset(Pmz_local_d, 0, nvox * sizeof(double)));
    CHECK_CUDA(cudaMemset(Psx_local_d, 0, nvox * sizeof(double)));
    CHECK_CUDA(cudaMemset(Psy_local_d, 0, nvox * sizeof(double)));
    CHECK_CUDA(cudaMemset(Psz_local_d, 0, nvox * sizeof(double)));

    CHECK(cudaMalloc((void**)&d_grain_mat, ng_total * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_grain_mat, 0, ng_total * sizeof(int)));
    grain_mat = (int*)malloc(ng_total * sizeof(int));
    
    CHECK(cudaMalloc(&d_is_interface, nvox * sizeof(int)));

}
