// Add this small helper up top (or just inline the formula)

#include "header.h"
#include "check.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h> 

__device__ inline double rand_normal_device(double mean, double sigma, curandState* st) {
    return mean + sigma * curand_normal_double(st);
}

__global__ void init_PmPs_by_material(
    double *Pmx, double *Pmy, double *Pmz,
    double *Psx, double *Psy, double *Psz,
    const int *omega, const int *grain_mat,
    const double *rand, int nx, int ny, int nz,
    double fe_noise, double afe_noise)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz) return;

    int id = ix + nx * (iy + iz * ny);

    int gid = omega[id];
    if (gid < 0) {
      // void/outside grain: set all to 0 and return
      Pmx[id]=Pmy[id]=Pmz[id]=0.0;
      Psx[id]=Psy[id]=Psz[id]=0.0;
      return;
    }
    int mat = grain_mat[gid];   // 0=AFE, 1=FE, 2=DE

    double r = 2.0 * rand[id] - 1.0;

    if (mat == 1) {
        // FE
        Pmx[id] = fe_noise * r;
        Pmy[id] = fe_noise * r;
        Pmz[id] = fe_noise * r;
        Psx[id] = 0.0;
        Psy[id] = 0.0;
        Psz[id] = 0.0;
    }
    else if (mat == 0) {
        // AFE
        Pmx[id] = 0.0;
        Pmy[id] = 0.0;
        Pmz[id] = 0.0;
        Psx[id] = 1.0;//afe_noise * r;
        Psy[id] = 1.0;//afe_noise * r;
        Psz[id] = 1.0;//afe_noise * r;
    }
    else {
        // DE
        Pmx[id] = 0.0;
        Pmy[id] = 0.0;
        Pmz[id] = 0.0;
        Psx[id] = 0.0;
        Psy[id] = 0.0;
        Psz[id] = 0.0;
    }
}

__global__ void InitConfKernel(
    double* alpha_m, double* beta_m, double* gamma_m, double* g_m, double* a0_m,
    double* kappa_nd_m, double* L_nd_m, double* eps_m, int nx, int ny, int nz,
    MaterialNorm N_AFE, MaterialNorm N_FE, MaterialNorm N_DE,
    int* grain_mat, int* omega,  unsigned long long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int id  = i + nx * (j + k * ny);
    int gid = omega[id];

    // ---- Guard against invalid omega entries
    if (gid < 0) return;                    // empty / outside grain
    // optionally: if (gid >= ng_total) return;  // needs ng_total on device (pass as param or __constant__)

    int mat_type = grain_mat[gid];

    curandState state;
    curand_init(seed, id, 0, &state);

    MaterialNorm N;
    if (mat_type == 2)      N = N_DE;
    else if (mat_type == 1) N = N_FE;
    else                    N = N_AFE;

    double sigma_frac = 0.0;

    alpha_m[id]    = rand_normal_device(N.alpha_nd,  sigma_frac * fabs(N.alpha_nd),  &state);
    beta_m[id]     = rand_normal_device(N.beta_nd,   sigma_frac * fabs(N.beta_nd),   &state);
    gamma_m[id]    = rand_normal_device(N.gamma_nd,  sigma_frac * fabs(N.gamma_nd),  &state);
    g_m[id]        = rand_normal_device(N.g_nd,      sigma_frac * fabs(N.g_nd),      &state);
    a0_m[id]       = rand_normal_device(N.a0_nd,     sigma_frac * fabs(N.a0_nd),     &state);
    kappa_nd_m[id] = N.kappa_nd;
    L_nd_m[id]     = N.L_nd;
    eps_m[id]      = N.eps_r;
}

void Init_Conf() {

	    printf("\n=== Init_Conf(): Assigning materials to grains ===\n");

    // -------------------------------------------------------------------
    // 0) Build layer z-ranges
    // Convention z = 0 is bottom and z = nz-1 is top surface
    // -------------------------------------------------------------------
    int *z_start = (int*)malloc(h_n_layers * sizeof(int));
    int *z_end   = (int*)malloc(h_n_layers * sizeof(int));

    z_start[0] = 0;
    z_end[0]   = h_layer_counts[0];

    for (int L = 1; L < h_n_layers; L++) {
        z_start[L] = z_end[L-1];
        z_end[L]   = z_start[L] + h_layer_counts[L];
    }

    // -------------------------------------------------------------------
    // 1) Assign each grain to a layer based on centroid z
    // -------------------------------------------------------------------
    int *grain_layer = (int*)malloc(ng_total * sizeof(int));

    for (int g = 0; g < ng_total; g++) {
        double zc = centz[g];
        int found = 0;

        for (int L = 0; L < h_n_layers; L++) {
            if (zc >= z_start[L] && zc < z_end[L]) {
                grain_layer[g] = L;
                found = 1;
                break;
            }
        }
        if (!found) {
            fprintf(stderr,
                "Init_Conf ERROR: grain %d (zc=%g) not found in any layer.\n",
                g, zc);
            exit(1);
        }
    }

    // -------------------------------------------------------------------
    // 2) Count how many voxels belong to each grain
    // -------------------------------------------------------------------
    int *vox_per_grain = (int*)calloc(ng_total, sizeof(int));

    for (int id = 0; id < nx*ny*nz; id++) {
        int g = omega[id];
        if (g >= 0 && g < ng_total)
            vox_per_grain[g]++;
    }

    // -------------------------------------------------------------------
    // 3) Grain material array
    // -------------------------------------------------------------------
    // initialize with -1 (unassigned)
    for (int g = 0; g < ng_total; g++)
        grain_mat[g] = -1;

    srand(123456);  // deterministic

    // -------------------------------------------------------------------
    // 4) Process every layer independently
    // -------------------------------------------------------------------
    for (int L = 0; L < h_n_layers; L++) {

        printf("\n-- Layer %d material assignment --\n", L);

        double A = afe_frac_layer[L];
        double F = fe_frac_layer[L];
        double D = de_frac_layer[L];

        printf("Requested fractions: AFE=%.3f FE=%.3f DE=%.3f\n", A, F, D);

        // normalize (safety)
        double sum = A + F + D;
        if (sum <= 0) { A = 0; F = 0; D = 1; sum = 1; }
        A /= sum; F /= sum; D /= sum;

        // -------------------------------------
        // 4A) Collect grains in this layer
        // -------------------------------------
        int *lst = (int*)malloc(ng_total * sizeof(int));
        int cnt = 0;

        for (int g = 0; g < ng_total; g++)
            if (grain_layer[g] == L)
                lst[cnt++] = g;

        printf("Layer %d contains %d grains\n", L, cnt);

        // -------------------------------------
        // 4B) Total voxels in this layer
        // -------------------------------------
        long long tot_vox = 0;
        for (int i = 0; i < cnt; i++)
            tot_vox += vox_per_grain[lst[i]];

        if (tot_vox <= 0) {
            fprintf(stderr, "Init_Conf ERROR: layer %d has zero voxels!\n", L);
            exit(1);
        }

        long long target_A = llround(A * tot_vox);
        long long target_F = llround(F * tot_vox);
        long long target_D = llround(D * tot_vox);

        printf("Voxel targets: AFE=%lld FE=%lld DE=%lld (tot=%lld)\n",
               target_A, target_F, target_D, tot_vox);

        // -------------------------------------
        // 4C) Randomize grain order
        // -------------------------------------
        for (int i = 0; i < cnt; i++) {
            int r = i + rand() % (cnt - i);
            int tmp = lst[i]; lst[i] = lst[r]; lst[r] = tmp;
        }

        // -------------------------------------
        // 4D) Assign DE first
        // -------------------------------------
        long long acc = 0;
        int p = 0;

        while (p < cnt && acc < target_D) {
            int g = lst[p++];
            grain_mat[g] = 2;  // DE
            acc += vox_per_grain[g];
        }

        // -------------------------------------
        // 4E) Assign AFE
        // -------------------------------------
        acc = 0;
        while (p < cnt && acc < target_A) {
            int g = lst[p++];
            grain_mat[g] = 0;  // AFE
            acc += vox_per_grain[g];
        }

        // -------------------------------------
        // 4F) Assign FE for all remaining grains
        // -------------------------------------
        while (p < cnt) {
            int g = lst[p++];
            if (grain_mat[g] == -1)
                grain_mat[g] = 1;  // FE
        }

        free(lst);
    }

    // -------------------------------------------------------------------
    // 5) Debug voxel fractions after assignment
    // -------------------------------------------------------------------
    size_t vA = 0, vF = 0, vD = 0, vT = (size_t)nx*ny*nz;

    for (int id = 0; id < nx*ny*nz; id++) {
        int g = omega[id];
        if (g < 0) continue;
        int mt = grain_mat[g];
        if (mt == 0) vA++;
        else if (mt == 1) vF++;
        else if (mt == 2) vD++;
    }

    printf("\nFinal voxel fractions:\n");
    printf("  AFE = %.3f %%\n", 100.0*vA/vT);
    printf("  FE  = %.3f %%\n", 100.0*vF/vT);
    printf("  DE  = %.3f %%\n", 100.0*vD/vT);

    // -------------------------------------------------------------------
    // 6) Copy grain_mat to GPU
    // -------------------------------------------------------------------
    CHECK(cudaMemcpy(d_grain_mat, grain_mat, ng_total * sizeof(int),
                     cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------
    // 7) Initialize polarization fields on GPU
    // -------------------------------------------------------------------

    dim3 threads(8,8,8);
    dim3 blocks((nx+7)/8, (ny+7)/8, (nz+7)/8);
    unsigned long long seed = (unsigned long long)time(NULL);

    InitConfKernel<<<blocks, threads>>>(alpha_m, beta_m, gamma_m,
                         g_m, a0_m, kappa_nd_m, L_nd_m, eps_m,
                         nx, ny, nz, N_AFE, N_FE, N_DE,
                         d_grain_mat, omega_d, seed);

   cudaDeviceSynchronize();

   int *grain_map_h = (int*)malloc(nx * ny * nz * sizeof(int));
 
   for (int id = 0; id < nx * ny * nz; id++) {
      int gid = omega[id];         // host copy of omega
      if (gid < 0) grain_map_h[id] = -1;   // void or outside
      else grain_map_h[id] = grain_mat[gid];
   }

   // ---- Noise seeding
   size_t nvox = (size_t)nx * ny * nz;
   double *rand_d = NULL;
   CHECK_CUDA(cudaMalloc((void **)&rand_d, nvox * sizeof(double)));

   curandGenerator_t gen;
   curandStatus_t st;
   st = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
   if (st != CURAND_STATUS_SUCCESS) {
      fprintf(stderr,"curandCreateGenerator failed\n");
      exit(1);
   }

   st = curandSetPseudoRandomGeneratorSeed(gen, 34635ULL);
   if (st != CURAND_STATUS_SUCCESS) {
      fprintf(stderr,"curandSetSeed failed\n");
      exit(1);
   }

   st = curandGenerateUniformDouble(gen, rand_d, nvox);
   if (st != CURAND_STATUS_SUCCESS) {
       fprintf(stderr,"curandGenerateUniformDouble failed (%d)\n", (int)st);
       exit(1);
   }

   // Launch correct material-dependent init kernel
   init_PmPs_by_material<<<blocks, threads>>>(Pmx_old_d, Pmy_old_d, Pmz_old_d,
                        Psx_old_d, Psy_old_d, Psz_old_d, omega_d, d_grain_mat,
                       rand_d, nx, ny, nz, 1e-1, PsNoise);
   CHECK_KERNEL();
   cudaDeviceSynchronize();

   curandDestroyGenerator(gen);
   CHECK_CUDA(cudaFree(rand_d));


   write_field("microstructure.txt", grain_map_h, nx, ny, nz, 1.0);
   free(grain_map_h);

   free(z_start);
   free(z_end);
   free(grain_layer);
   free(vox_per_grain);


   /*
// Allocate host buffers
double *Pmx_h = (double*)malloc(nvox * sizeof(double));
double *Pmy_h = (double*)malloc(nvox * sizeof(double));
double *Pmz_h = (double*)malloc(nvox * sizeof(double));

double *Psx_h = (double*)malloc(nvox * sizeof(double));
double *Psy_h = (double*)malloc(nvox * sizeof(double));
double *Psz_h = (double*)malloc(nvox * sizeof(double));

// Copy from device to host
CHECK(cudaMemcpy(Pmx_h, Pmx_old_d, nvox * sizeof(double), cudaMemcpyDeviceToHost));
CHECK(cudaMemcpy(Pmy_h, Pmy_old_d, nvox * sizeof(double), cudaMemcpyDeviceToHost));
CHECK(cudaMemcpy(Pmz_h, Pmz_old_d, nvox * sizeof(double), cudaMemcpyDeviceToHost));

CHECK(cudaMemcpy(Psx_h, Psx_old_d, nvox * sizeof(double), cudaMemcpyDeviceToHost));
CHECK(cudaMemcpy(Psy_h, Psy_old_d, nvox * sizeof(double), cudaMemcpyDeviceToHost));
CHECK(cudaMemcpy(Psz_h, Psz_old_d, nvox * sizeof(double), cudaMemcpyDeviceToHost));
//for(int i=0; i<100;i++) printf("%lf\n", Pmz_h[i]);
// ---------------- Write binary files ----------------
write_vector_field_binary("P_macro.bin", Pmx_h, Pmy_h, Pmz_h, nx, ny, nz, P0_ref);
write_vector_field_binary("P_spont.bin", Psx_h, Psy_h, Psz_h, nx, ny, nz, P0_ref);

// ---------------- Free host memory ----------------
free(Pmx_h); free(Pmy_h); free(Pmz_h);
free(Psx_h); free(Psy_h); free(Psz_h);
printf("Done\n");
exit(0);
*/

}
