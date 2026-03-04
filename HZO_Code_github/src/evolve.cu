#include "header.h"
#include "check.h"
#include "kernels_evol.h"     // NOT ../include/...
#include <cub/cub.cuh>        // needed for cub::DeviceReduce::Sum
#include <signal.h>
#include <stdlib.h>
#include <string.h>

// -----------------------------------------------------------------------------
// Resource helpers (unchanged)
// -----------------------------------------------------------------------------
void allocate_host_buffers() {
    if (Pmx == NULL && out_flag == 1) {
        size_t size = sizeof(double) * nx * ny * nz;
        Pmx = (double *)malloc(size);
        Pmy = (double *)malloc(size);
        Pmz = (double *)malloc(size);
        Psx = (double *)malloc(size);
        Psy = (double *)malloc(size);
        Psz = (double *)malloc(size);
        if (!Pmx || !Pmy || !Pmz || !Psx || !Psy || !Psz) {
            printf("Host malloc failed\n");
            exit(1);
        }
    }
}

void allocate_cuda_resources() {
    if (Psum_d == NULL) {
        CHECK(cudaMalloc((void**)&Psum_d, sizeof(double)));
    }
    if (t_storage == NULL) {
        cub::DeviceReduce::Sum(nullptr, t_storage_bytes, Pmz_old_d, Psum_d, nx * ny * nz);
        CHECK(cudaMalloc(&t_storage, t_storage_bytes));
    }
}

static bool g_events_created = false;

void create_cuda_events() {
    if (!g_events_created) {
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        g_events_created = true;
    }
}

void destroy_cuda_events() {
    if (g_events_created) {
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        g_events_created = false;
    }
}

void free_host_buffers() {
    if (Pmx) free(Pmx);
    if (Pmy) free(Pmy);
    if (Pmz) free(Pmz);
    if (Psx) free(Psx);
    if (Psy) free(Psy);
    if (Psz) free(Psz);

    Pmx = Pmy = Pmz = NULL;
    Psx = Psy = Psz = NULL;
}

void free_cuda_resources() {
    if (t_storage) {
        CHECK(cudaFree(t_storage));
        t_storage = NULL;
    }
    if (Psum_d) {
        CHECK(cudaFree(Psum_d));
        Psum_d = NULL;
    }
}

void evolve_resources_init() {
    allocate_host_buffers();
    allocate_cuda_resources();
    create_cuda_events();
}

void evolve_resources_cleanup() {
    free_host_buffers();
    free_cuda_resources();
    destroy_cuda_events();
}

void Evolve() {

    printf("Now in evolve.\n");
    int out_row = 0;
    const size_t Nvox     = (size_t)nx * ny * nz;
    int ndiv = 6, incr = 0, time_to_output;
    double dimE = 0.0;

    double Psum_global_m, Psum_global_s, Psum_local_m, Psum_local_s;
    double Pmean_global_m = 0.0, Pmean_global_s = 0.0;
    double Pmean_local_m  = 0.0, Pmean_local_s  = 0.0;

    double Psum_FE_m = 0.0;
    double sum_mask_FE = 0.0;
    double Pmean_FE_only = 0.0;
    double f_FE = 0.0;

    double *Pmz_FE_d = nullptr;
    double *mask_FE_d = nullptr;   // 1.0 for FE voxels, 0.0 otherwise

    CHECK(cudaMalloc(&Pmz_FE_d,  Nvox * sizeof(double)));
    CHECK(cudaMalloc(&mask_FE_d, Nvox * sizeof(double)));

    double Psum_DE_m = 0.0;
    double sum_mask_DE = 0.0;
    double Pmean_DE_only = 0.0;
    double f_DE = 0.0;

    double *Pmz_DE_d  = nullptr;
    double *mask_DE_d = nullptr;

    CHECK(cudaMalloc(&Pmz_DE_d,  Nvox * sizeof(double)));
    CHECK(cudaMalloc(&mask_DE_d, Nvox * sizeof(double)));

    double Psum_AFE_m = 0.0;
    double sum_mask_AFE = 0.0;
    double Pmean_AFE_only = 0.0;
    double f_AFE = 0.0;

    double *Pmz_AFE_d  = nullptr;
    double *mask_AFE_d = nullptr;

    CHECK(cudaMalloc(&Pmz_AFE_d,  Nvox * sizeof(double)));
    CHECK(cudaMalloc(&mask_AFE_d, Nvox * sizeof(double)));

    const double Ez_scale = E0_ref / 1.0e+08;    // output in MV/cm
    const double P0_scale = P0_ref * 100.0;

    double Psum = 0.0;


    incEz = delEz;


    char sweepfile[PATHBUF];
    //sim_index = 2;

    safe_snprintf(sweepfile, sizeof(sweepfile), "%s/sweep%03d.txt", output_dir, sim_index); 

    fvoln = fopen(sweepfile, "a");
    if (!fvoln) {
        printf("Error opening %s\n", sweepfile);
        exit(1);
    }

    time_to_output = time_to_change;

    int noise_done = 0;
    size_t nvox = (size_t)nx * ny * nz;

    // ----------------- CURAND stuff (only if using noise) -----------------
    double *rand_d = NULL;
    curandGenerator_t gen;
    if (strcmp(crystal_type, "single") == 0) { 
        CHECK_CUDA(cudaMalloc((void **)&rand_d, nvox * sizeof(double)));

        curandStatus_t st;
        st = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        if (st != CURAND_STATUS_SUCCESS) {
            fprintf(stderr,"curandCreateGenerator failed\n");
            exit(1);
        }

        st = curandSetPseudoRandomGeneratorSeed(gen, 98765ULL);
        if (st != CURAND_STATUS_SUCCESS) {
            fprintf(stderr,"curandSetSeed failed\n");
            exit(1);
        }
    }
    // ----------------------------------------------------------------------

    int steps_since_change = 0;
    int base_relax_steps   = time_to_change;

    int zero_relax_factor  = 10;

    cudaEventRecord(start, 0);

    // ---- Stage A: pre-relax at E=0 (optional) ----
    if (do_pre_relax) {
        Ez_ext = 0.0;
        int N_relax = 15000;   // or larger

        for (int step = 0; step < N_relax; ++step) {
            sim_time += del_t;

            kernelCalc_antiferro<<<blocks, threads>>>(
                Pmx_d, Pmx_old_d, Psx_d, Psx_old_d, Pmy_d, Pmy_old_d, Psy_d, Psy_old_d,
                Pmz_d, Pmz_old_d, Psz_d, Psz_old_d, omega_d, TR_d,
                alpha_m, beta_m, gamma_m, g_m, a0_m, kappa_nd_m, L_nd_m,
                del_h, del_t, delecX_d, delecY_d, delecZ_d,
                dep_flag, nx, ny, nz, ng_total, step, Ez_ext,
                g_top, g_bot, Pmx_local_d, Psx_local_d,
                Pmy_local_d, Psy_local_d, Pmz_local_d, Psz_local_d, d_is_interface);
            CHECK_KERNEL();

            kernelUpdate_P<<<numBlocks, threadsPerBlock>>>(
                Pmx_d, Pmx_old_d, Pmy_d, Pmy_old_d, Pmz_d, Pmz_old_d, nx, ny, nz);
            CHECK_KERNEL();

            kernelUpdate_P<<<numBlocks, threadsPerBlock>>>(
                Psx_d, Psx_old_d, Psy_d, Psy_old_d, Psz_d, Psz_old_d, nx, ny, nz);
            CHECK_KERNEL();

            if (dep_flag != 0)
                Depol_field();

            count++;   // optional; only needed if you care about time index
        }

        size_t size = sizeof(double) * nx * ny * nz;
        double *testv1 = (double *)malloc(size);

        CHECK(cudaMemcpy(testv1, Psz_old_d, nvox * sizeof(double), cudaMemcpyDeviceToHost));
        write_midplane_slice("slice_E0_midplane.dat", testv1, nx, ny, nz);

        CHECK(cudaMemcpy(testv1, Psz_local_d, nvox * sizeof(double), cudaMemcpyDeviceToHost));
        write_midplane_slice("slice_E0_midplane_local.dat", testv1, nx, ny, nz);
        free(testv1);
    }

    // ---- Stage B: main sweeping loop ----


int stride = 4;  

char field_dir[PATHBUF];
safe_snprintf(field_dir, sizeof(field_dir), "%s/sim%03d", output_dir, sim_index);
make_dir_p(field_dir);

    Ez_prev = Ez_ext;

    while (peloop < 2) {
        sim_time += del_t;

        kernelCalc_antiferro<<<blocks, threads>>>(
            Pmx_d, Pmx_old_d, Psx_d, Psx_old_d, Pmy_d, Pmy_old_d, Psy_d, Psy_old_d,
            Pmz_d, Pmz_old_d, Psz_d, Psz_old_d, omega_d, TR_d, alpha_m, beta_m, gamma_m, g_m, a0_m,
            kappa_nd_m, L_nd_m, del_h, del_t, delecX_d, delecY_d, delecZ_d, dep_flag, 
	    nx, ny, nz, ng_total, count, Ez_ext,  g_top, g_bot,
            Pmx_local_d, Psx_local_d, Pmy_local_d, Psy_local_d, Pmz_local_d, Psz_local_d , d_is_interface);
        CHECK_KERNEL();

        kernelUpdate_P<<<numBlocks, threadsPerBlock>>>(
            Pmx_d, Pmx_old_d, Pmy_d, Pmy_old_d, Pmz_d, Pmz_old_d, nx, ny, nz);
        CHECK_KERNEL();

        kernelUpdate_P<<<numBlocks, threadsPerBlock>>>(
            Psx_d, Psx_old_d, Psy_d, Psy_old_d, Psz_d, Psz_old_d, nx, ny, nz);
        CHECK_KERNEL();

        if (dep_flag != 0)
            Depol_field();

        steps_since_change++;
        count++;

           double E_MVcm = Ez_ext * Ez_scale;
           double Ewin_nd = (0.2e8) / E0_ref;
        // --------------------- relaxation ---------------------
        int relax_steps_here = base_relax_steps;
        if ( (strcmp(crystal_type, "single") == 0 )&& (fabs(Ez_ext) < Ewin_nd)) {
            // near E≈0 → extra relax only if flag ON
            relax_steps_here = base_relax_steps * zero_relax_factor;
        }
        // -----------------------------------------------------------

        // ==========================================================

        if (steps_since_change >= relax_steps_here) {

	   double E_prev_MVcm = Ez_prev * Ez_scale;

if (peloop > 0 && out_flag == 1) {

    out_row++;  // 1 for first written row (e.g., 18400), 2 for next (19200), ...

    if ((out_row % stride) == 1) {   // dumps row 1, 1+stride, 1+2*stride, ...
        char fnamePm[PATHBUF], fnamePs[PATHBUF];

        safe_snprintf(fnamePm, sizeof(fnamePm), "%s/Pm_%06d.bin", field_dir, count);
        safe_snprintf(fnamePs, sizeof(fnamePs), "%s/Ps_%06d.bin", field_dir, count);

        CHECK(cudaMemcpy(Pmx, Pmx_old_d, nvox*sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(Psx, Psx_old_d, nvox*sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(Pmy, Pmy_old_d, nvox*sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(Psy, Psy_old_d, nvox*sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(Pmz, Pmz_old_d, nvox*sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(Psz, Psz_old_d, nvox*sizeof(double), cudaMemcpyDeviceToHost));

        write_vector_field_binary(fnamePm, Pmx, Pmy, Pmz, nx, ny, nz, P0_scale);
        write_vector_field_binary(fnamePs, Psx, Psy, Psz, nx, ny, nz, P0_scale);
    }
}


           // global Pm_z (lab-frame)
           cub::DeviceReduce::Sum(t_storage, t_storage_bytes, Pmz_old_d, Psum_d, (int)Nvox);
           CHECK(cudaMemcpy(&Psum_global_m, Psum_d, sizeof(double), cudaMemcpyDeviceToHost));
           Pmean_global_m = Psum_global_m * one_by_nxnynz;

           // global Ps_z (lab-frame)
           cub::DeviceReduce::Sum(t_storage, t_storage_bytes, Psz_old_d, Psum_d, (int)Nvox);
           CHECK(cudaMemcpy(&Psum_global_s, Psum_d, sizeof(double), cudaMemcpyDeviceToHost));
           Pmean_global_s = Psum_global_s * one_by_nxnynz;

           // local Pm_z (crystal frame)
           cub::DeviceReduce::Sum(t_storage, t_storage_bytes, Pmz_local_d, Psum_d, (int)Nvox);
           CHECK(cudaMemcpy(&Psum_local_m, Psum_d, sizeof(double), cudaMemcpyDeviceToHost));
           Pmean_local_m = Psum_local_m * one_by_nxnynz;

           // local Ps_z (crystal frame)
           cub::DeviceReduce::Sum(t_storage, t_storage_bytes, Psz_local_d, Psum_d, (int)Nvox);
           CHECK(cudaMemcpy(&Psum_local_s, Psum_d, sizeof(double), cudaMemcpyDeviceToHost));
           Pmean_local_s = Psum_local_s * one_by_nxnynz;

           initcount += 1;

	   // ---------- FE-only polarization stats (lab-frame Pm_z) ----------
	   {
               int t = 256;
               int b = (int)((Nvox + (size_t)t - 1) / (size_t)t);

	       build_mat_mask_and_p<<<b, t>>>(Pmz_old_d, omega_d, d_grain_mat,
                               Pmz_FE_d, mask_FE_d, (int)Nvox, 1 /*FE*/);

               CHECK_KERNEL();

               // sum masked P over FE voxels
               cub::DeviceReduce::Sum(t_storage, t_storage_bytes, Pmz_FE_d, Psum_d, (int)Nvox);
               CHECK(cudaMemcpy(&Psum_FE_m, Psum_d, sizeof(double), cudaMemcpyDeviceToHost));

               // sum FE mask (counts FE voxels)
               cub::DeviceReduce::Sum(t_storage, t_storage_bytes, mask_FE_d, Psum_d, (int)Nvox);
               CHECK(cudaMemcpy(&sum_mask_FE, Psum_d, sizeof(double), cudaMemcpyDeviceToHost));

               if (sum_mask_FE > 0.0) {
                  Pmean_FE_only = Psum_FE_m / sum_mask_FE;     // FE-only mean Pm_z
                  f_FE          = sum_mask_FE / (double)Nvox;  // FE voxel fraction
               } else {
                 Pmean_FE_only = 0.0;
                 f_FE          = 0.0;
               }
            }
            // ---------------------------------------------------------------

	   // ---------- DE-only polarization stats (lab-frame Pm_z) ----------
           {
	       int t = 256;
	       int b = (int)((Nvox + (size_t)t - 1) / (size_t)t);

               build_mat_mask_and_p<<<b, t>>>(Pmz_old_d, omega_d, d_grain_mat,
                                   Pmz_DE_d, mask_DE_d, (int)Nvox, 2 /*DE*/);
               CHECK_KERNEL();

               cub::DeviceReduce::Sum(t_storage, t_storage_bytes, Pmz_DE_d, Psum_d, (int)Nvox);
               CHECK(cudaMemcpy(&Psum_DE_m, Psum_d, sizeof(double), cudaMemcpyDeviceToHost));

	       cub::DeviceReduce::Sum(t_storage, t_storage_bytes, mask_DE_d, Psum_d, (int)Nvox);
	       CHECK(cudaMemcpy(&sum_mask_DE, Psum_d, sizeof(double), cudaMemcpyDeviceToHost));

	       if (sum_mask_DE > 0.0) {
	          Pmean_DE_only = Psum_DE_m / sum_mask_DE;
	  	  f_DE          = sum_mask_DE / (double)Nvox;
	       } else {
	          Pmean_DE_only = 0.0;
                  f_DE = 0.0;
               }
	   }
	   // ---------------------------------------------------------------

	   // ---------- AFE-only polarization stats (lab-frame Pm_z) ----------
           {
	       int t = 256;
       	       int b = (int)((Nvox + (size_t)t - 1) / (size_t)t);

       	       build_mat_mask_and_p<<<b, t>>>(Pmz_old_d, omega_d, d_grain_mat,
                                   Pmz_AFE_d, mask_AFE_d, (int)Nvox,
                                   0 /*AFE*/);   // <-- set correct AFE ID here

    	       CHECK_KERNEL();

	       cub::DeviceReduce::Sum(t_storage, t_storage_bytes, Pmz_AFE_d, Psum_d, (int)Nvox);
    	       CHECK(cudaMemcpy(&Psum_AFE_m, Psum_d, sizeof(double), cudaMemcpyDeviceToHost));

	       cub::DeviceReduce::Sum(t_storage, t_storage_bytes, mask_AFE_d, Psum_d, (int)Nvox);
	       CHECK(cudaMemcpy(&sum_mask_AFE, Psum_d, sizeof(double), cudaMemcpyDeviceToHost));

	       if (sum_mask_AFE > 0.0) {
       	            Pmean_AFE_only = Psum_AFE_m / sum_mask_AFE;
                    f_AFE = sum_mask_AFE / (double)Nvox;
	       } else {
	            Pmean_AFE_only = 0.0;
                    f_AFE = 0.0;
               }
	   }
	   // ---------------------------------------------------------------


           // hysteresis direction logic
           if (Ez_ext > maxEz) {
	               // We are at the positive turning point (overshoot, e.g. 3.06 MV/cm)
               //if (out_flag == 2 && peloop > 0 && !saved_Emax) {
               //    char tag_sim[64];
               //    sprintf(tag_sim, "Emax_sim%03d", sim_index);
               //    save_snapshot(tag_sim, Pmx_old_d, Pmy_old_d, Pmz_old_d,
               //           Psx_old_d, Psy_old_d, Psz_old_d, P0_scale);
               //    saved_Emax = true;
               //    printf("SNAPSHOT Emax: count=%d, E=%g MV/cm\n", count, E_MVcm);
               //}
               incEz  = -delEz;
               peloop += 1;
           }
           if (Ez_ext < -maxEz) {
		           // Negative turning point (overshoot, e.g. -3.06 MV/cm)
              if (out_flag == 2 && peloop > 0 && !saved_Emin) {
                  char tag_sim[64];
                  sprintf(tag_sim, "Emin_sim%03d", sim_index);
                  save_snapshot(tag_sim, Pmx_old_d, Pmy_old_d, Pmz_old_d, Psx_old_d, 
				  Psy_old_d, Psz_old_d, P0_scale);
                  saved_Emin = true;
                  printf("SNAPSHOT Emin: count=%d, E=%g MV/cm\n", count, E_MVcm);
              }
                incEz      = delEz;
                noise_done = 0;
           }

           // ------------- Ps noise injection (only if flag ON) -------------
           if (strcmp(crystal_type, "single") == 0 && peloop > 0 && !noise_done && fabs(Ez_ext) < Ewin_nd) {

                double noise_level = 1.0e-04;

                curandStatus_t st2 = curandGenerateUniformDouble(gen, rand_d, nvox);
                if (st2 != CURAND_STATUS_SUCCESS) {
                    fprintf(stderr,"curandGenerateUniformDouble failed (%d)\n", (int)st2);
                    exit(1);
                }

                kernelAddSmallNoise_Ps<<<blocks, threads>>>(rand_d, Psz_old_d, noise_level, nx, ny, nz);
                CHECK_KERNEL();

                noise_done = 1;
                printf("Injected small Ps noise at E = %g MV/cm (step %d)\n", E_MVcm, count);
           }
           // -----------------------------------------------------------------

	   if (out_flag == 2 && peloop > 0 && !saved_E0 && (E_prev_MVcm > 0.0 && E_MVcm <= 0.0)) {
               char tag_sim[64];
               sprintf(tag_sim, "E0_sim%03d", sim_index);
               save_snapshot(tag_sim, Pmx_old_d, Pmy_old_d, Pmz_old_d,
                      Psx_old_d, Psy_old_d, Psz_old_d,   P0_scale);
               saved_E0 = true;
               printf("SNAPSHOT E0: count=%d, E=%g MV/cm (prev=%g)\n", count, E_MVcm, E_prev_MVcm);
           }

           if (peloop > 0) {
              fprintf(fvoln, "%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%d\n",
                    Ez_ext * Ez_scale,
                    Pmean_global_m * P0_scale,
                    Pmean_global_s * P0_scale,
                    Pmean_local_m  * P0_scale,
                    Pmean_local_s  * P0_scale,
                    Pmean_FE_only  * P0_scale,  f_FE,
                    Pmean_AFE_only * P0_scale,  f_AFE,
                    Pmean_DE_only  * P0_scale,  f_DE,
                        count);

		   printf("%lf, %lf\n", Ez_ext * Ez_scale, Pmean_global_m * P0_scale);
           }

	    Ez_prev = Ez_ext;   // store current field

            Ez_ext += incEz;
            steps_since_change = 0;
        }
    }

    fclose(fvoln);

    // Clean up CURAND only if we created it
    if (strcmp(crystal_type, "single") == 0) {
        curandDestroyGenerator(gen);
        cudaFree(rand_d);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel execution time: %f s\n", elapsedTime / 1000.0);

    fflush(stdout);

    if (Pmz_FE_d)  cudaFree(Pmz_FE_d);
    if (mask_FE_d) cudaFree(mask_FE_d);

    if (Pmz_DE_d)  cudaFree(Pmz_DE_d);
    if (mask_DE_d) cudaFree(mask_DE_d);
    
    if (Pmz_AFE_d)  cudaFree(Pmz_AFE_d);
    if (mask_AFE_d) cudaFree(mask_AFE_d);


}
