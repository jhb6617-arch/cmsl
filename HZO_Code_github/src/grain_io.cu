#include "header.h"          // types.h + globals.h (+ cuda_deps.h under nvcc)
#include "check.h"           // CHECK / CHECK_CUDA / CHECK_KERNEL macros
#include "io_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


void ReadGrainCount(const char *dir, const char *grain_tag)
{
    char path_gnumb[256];
    snprintf(path_gnumb, sizeof(path_gnumb), "%s/gnumb_%s.txt", dir, grain_tag);

    FILE *finit = fopen(path_gnumb, "r");
    if (!finit) { perror("Error opening grain number file"); exit(1); }

    if (fscanf(finit, "%d", &ng_total) != 1) {
        fprintf(stderr, "Failed to read number of grains\n");
        exit(1);
    }
    fclose(finit);

    printf("Num of grains read = %d\n", ng_total);
}
void LoadGrainStructureFromDir(const char *dir, const char *grain_tag)
{
    FILE *finit;

    size_t ntot = (size_t)nx * ny * nz;

    // Allocate temporary host buffers
    int    *omega_f = (int*)    malloc(ntot * sizeof(int));
    double *eta_f   = (double*) malloc(ntot * sizeof(double));
    if (!omega_f || !eta_f) { fprintf(stderr, "malloc failed\n"); exit(1); }

    // reset host arrays (omega/eta already allocated by AllocateData)
    for (size_t i = 0; i < ntot; i++) {
        omega[i] = -1;
        eta[i]   = 0.0;
    }
    for (int i = 0; i < ng_total; i++) {
        centx[i] = centy[i] = centz[i] = -1.0;
        phi[i] = theta[i] = psi[i] = 0.0;
    }

    // --- Read grain structure binary ---
    char path_gdata[256];
    snprintf(path_gdata, sizeof(path_gdata), "%s/gdata_%s.bin", dir, grain_tag);

    finit = fopen(path_gdata, "rb");
    if (!finit) { perror(path_gdata); exit(1); }

    if (fread(omega_f, sizeof(int), ntot, finit) != ntot ||
        fread(eta_f,   sizeof(double), ntot, finit) != ntot) {
        fprintf(stderr, "Error reading %s\n", path_gdata);
        exit(1);
    }
    fclose(finit);

    // Convert from Fortran to C ordering
    for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
    for (int k = 0; k < nz; k++) {
        int id_f = k + nz * (j + i * ny); // Fortran
        int id_c = i + nx * (j + k * ny); // C
        omega[id_c] = omega_f[id_f];
        eta[id_c]   = eta_f[id_f];
    }

    free(omega_f);
    free(eta_f);

    // --- Read grain characteristics ---
    char path_gchar[256];
    snprintf(path_gchar, sizeof(path_gchar), "%s/gchar_%s.txt", dir, grain_tag);

    finit = fopen(path_gchar, "r");
    if (!finit) { perror(path_gchar); exit(1); }

    int gnum;
    double xcent, ycent, zcent;
    double gphi, gtheta, gpsi;

    for (int k = 0; k < ng_total; k++) {
        if (fscanf(finit, "%d %lf %lf %lf %lf %lf %lf",
                   &gnum, &xcent, &ycent, &zcent,
                   &gphi, &gtheta, &gpsi) != 7) {
            fprintf(stderr, "Error reading %s line %d\n", path_gchar, k);
            exit(1);
        }
        // assumes gnum in [0, ng_total-1]
        centx[gnum] = xcent;
        centy[gnum] = ycent;
        centz[gnum] = zcent;
        phi[gnum]   = gphi;
        theta[gnum] = gtheta;
        psi[gnum]   = gpsi;
    }
    fclose(finit);

    // Build TR_h
    for (int kp = 0; kp < ng_total; kp++) {
        double ph = phi[kp], th = theta[kp], ps = psi[kp];

        double sinph = sin(ph),   cosph = cos(ph);
        double sinth = sin(th),   costh = cos(th);
        double sinps = sin(ps),   cosps = cos(ps);

        double cth_sph = costh * sinph;
        double cth_cph = costh * cosph;
        double sth_sph = sinth * sinph;
        double sth_cph = sinth * cosph;

        TR_h[kp*9+0] = cosph*cosps - cth_sph*sinps;
        TR_h[kp*9+1] = sinph*cosps + cth_cph*sinps;
        TR_h[kp*9+2] = sinth*sinps;

        TR_h[kp*9+3] = -cosph*sinps - cth_sph*cosps;
        TR_h[kp*9+4] = -sinph*sinps + cth_cph*cosps;
        TR_h[kp*9+5] = sinth*cosps;

        TR_h[kp*9+6] = sth_sph;
        TR_h[kp*9+7] = -sth_cph;
        TR_h[kp*9+8] = costh;
    }

    // Copy to GPU
    CHECK(cudaMemcpy(phi_d,   phi,   sizeof(double)*ng_total,       cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(theta_d, theta, sizeof(double)*ng_total,       cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(psi_d,   psi,   sizeof(double)*ng_total,       cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(TR_d,    TR_h,  sizeof(double)*9*ng_total,     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(eta_d,   eta,   sizeof(double)*ntot,           cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(omega_d, omega, sizeof(int)*ntot,              cudaMemcpyHostToDevice));
}

