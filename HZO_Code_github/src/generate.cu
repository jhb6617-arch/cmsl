// Write combined vector field (3 components in order)

#include "header.h"
#include "check.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h> 


void write_vector_field_binary(const char* filename, double *Px, double *Py, double *Pz, int nx, int ny, int nz, double p0) {
           FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(1);
    }

    int N = nx * ny * nz;
    // Write all three components interleaved: (vx, vy, vz) for each point
    for (int i = 0; i < N; i++) {
        double x = Px[i] * p0;
        double y = Py[i] * p0;
        double z = Pz[i] * p0;
        fwrite(&x, sizeof(double), 1, f);
        fwrite(&y, sizeof(double), 1, f);
        fwrite(&z, sizeof(double), 1, f);
    }

    fclose(f);
}


void write_field(const char *filename, int *Pmz, int nx, int ny, int nz, double p0){
        /*
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(1);
    }

    int N = nx * ny * nz;
    // Write all three components interleaved: (vx, vy, vz) for each point
    for (int k = 0; k < N; k++) {
        double z = Pmz[k] * p0;
        fwrite(&z, sizeof(double), 1, f);
    }

    fclose(f);
*/

    FILE *f = fopen(filename, "w");
    for (int i=0; i<nx; i++){
     for (int j=0; j<ny; j++){
        for (int k=0; k<nz; k++){
          int id_c = i + nx * (j + k * ny);
          int h  = Pmz[id_c] * p0;
           fprintf(f,"%d\t%d\t%d\t%d\n", i, j, k, h);
        }
      fprintf(f,"\n");
     } fprintf(f,"\n");
    }

    fclose(f);

}

void write_midplane_slice(const char *fname,
                          const double *Psz_host,
                          int nx, int ny, int nz)
{
    FILE *f = fopen(fname, "w");
    if (!f) {
        fprintf(stderr, "Cannot open %s for writing slice\n", fname);
        return;
    }

    int k_mid = nz / 2;  // middle plane
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int id = i + nx * (j + k_mid * ny);
            // write i, j, value
            fprintf(f, "%d %d %.16e\n", i, j, Psz_host[id]);
        }
           fprintf(f, "\n"); // blank line between rows (gnuplot likes it for 'matrix' style)
     }
           
      fclose(f);
}    

void write_checkpoint(const char *fname,
                      double *Pmx_old_d, double *Pmy_old_d, double *Pmz_old_d,
                      double *Psx_old_d, double *Psy_old_d, double *Psz_old_d,
                      double *Ex_d, double *Ey_d, double *Ez_d, // or delecX_d etc.
                      int nx, int ny, int nz,
                      double Ez_ext, int peloop, int count, double incEz)
{
    size_t Nvox = (size_t)nx * ny * nz;

    double *Pmx = (double*)malloc(Nvox * sizeof(double));
    double *Pmy = (double*)malloc(Nvox * sizeof(double));
    double *Pmz = (double*)malloc(Nvox * sizeof(double));
    double *Psx = (double*)malloc(Nvox * sizeof(double));
    double *Psy = (double*)malloc(Nvox * sizeof(double));
    double *Psz = (double*)malloc(Nvox * sizeof(double));
    double *Ex  = (double*)malloc(Nvox * sizeof(double));
    double *Ey  = (double*)malloc(Nvox * sizeof(double));
    double *Ez  = (double*)malloc(Nvox * sizeof(double));

    FILE  *f    = NULL;             
    size_t N    = Nvox;             

    if (!Pmx || !Pmy || !Pmz || !Psx || !Psy || !Psz || !Ex || !Ey || !Ez) {
        fprintf(stderr, "Checkpoint malloc failed\n");
        goto cleanup;
    }

    CHECK(cudaMemcpy(Pmx, Pmx_old_d, Nvox * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(Pmy, Pmy_old_d, Nvox * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(Pmz, Pmz_old_d, Nvox * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK(cudaMemcpy(Psx, Psx_old_d, Nvox * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(Psy, Psy_old_d, Nvox * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(Psz, Psz_old_d, Nvox * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK(cudaMemcpy(Ex,  Ex_d,      Nvox * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(Ey,  Ey_d,      Nvox * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(Ez,  Ez_d,      Nvox * sizeof(double), cudaMemcpyDeviceToHost));

    f = fopen(fname, "wb");
    if (!f) {
        fprintf(stderr, "Cannot open %s for writing\n", fname);
        goto cleanup;
    }

    fwrite(&nx,     sizeof(int),    1, f);
    fwrite(&ny,     sizeof(int),    1, f);
    fwrite(&nz,     sizeof(int),    1, f);
    fwrite(&peloop, sizeof(int),    1, f);
    fwrite(&count,  sizeof(int),    1, f);
    fwrite(&Ez_ext, sizeof(double), 1, f);
    fwrite(&incEz,  sizeof(double), 1, f);   // <--- NEW

    fwrite(Pmx, sizeof(double), N, f);
    fwrite(Pmy, sizeof(double), N, f);
    fwrite(Pmz, sizeof(double), N, f);
    fwrite(Psx, sizeof(double), N, f);
    fwrite(Psy, sizeof(double), N, f);
    fwrite(Psz, sizeof(double), N, f);
    fwrite(Ex,  sizeof(double), N, f);
    fwrite(Ey,  sizeof(double), N, f);
    fwrite(Ez,  sizeof(double), N, f);

    printf("Checkpoint written to %s (peloop=%d, count=%d, Ez_ext=%g, incEz=%g)\n",
           fname, peloop, count, Ez_ext, incEz);

cleanup:
    if (f) fclose(f);

    free(Pmx);
    free(Pmy);
    free(Pmz);
    free(Psx);
    free(Psy);
    free(Psz);
    free(Ex);
    free(Ey);
    free(Ez);
}

void read_checkpoint(const char *fname,
                     double *Pmx_old_d, double *Pmy_old_d, double *Pmz_old_d,
                     double *Psx_old_d, double *Psy_old_d, double *Psz_old_d,
                     double *Ex_d, double *Ey_d, double *Ez_d,
                     int nx, int ny, int nz,
                     int *peloop_out, int *count_out,
                     double *Ez_ext_out, double *incEz_out)
{
    size_t Nvox = (size_t)nx * ny * nz;

    double *Pmx = (double*)malloc(Nvox * sizeof(double));
    double *Pmy = (double*)malloc(Nvox * sizeof(double));
    double *Pmz = (double*)malloc(Nvox * sizeof(double));
    double *Psx = (double*)malloc(Nvox * sizeof(double));
    double *Psy = (double*)malloc(Nvox * sizeof(double));
    double *Psz = (double*)malloc(Nvox * sizeof(double));
    double *Ex  = (double*)malloc(Nvox * sizeof(double));
    double *Ey  = (double*)malloc(Nvox * sizeof(double));
    double *Ez  = (double*)malloc(Nvox * sizeof(double));
    
    FILE  *f    = NULL;             // declare before any goto
    size_t N    = Nvox;             // same

    f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s for reading\n", fname);
        goto cleanup;
    }

    int nx_chk, ny_chk, nz_chk;
    if (fread(&nx_chk, sizeof(int), 1, f) != 1 ||
        fread(&ny_chk, sizeof(int), 1, f) != 1 ||
        fread(&nz_chk, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Checkpoint header read failed (dims)\n");
        goto cleanup;
    }

    if (nx_chk != nx || ny_chk != ny || nz_chk != nz) {
        fprintf(stderr,
                "Checkpoint grid mismatch: file (%d,%d,%d), code (%d,%d,%d)\n",
                nx_chk, ny_chk, nz_chk, nx, ny, nz);
        goto cleanup;
    }

    if (fread(peloop_out, sizeof(int), 1, f) != 1 ||
        fread(count_out,  sizeof(int), 1, f) != 1 ||
        fread(Ez_ext_out, sizeof(double), 1, f) != 1 ||
        fread(incEz_out,  sizeof(double), 1, f) != 1) {
        fprintf(stderr, "Checkpoint header read failed (state)\n");
        goto cleanup;
    }

    N = Nvox;

    if (fread(Pmx, sizeof(double), N, f) != N ||
        fread(Pmy, sizeof(double), N, f) != N ||
        fread(Pmz, sizeof(double), N, f) != N ||
        fread(Psx, sizeof(double), N, f) != N ||
        fread(Psy, sizeof(double), N, f) != N ||
        fread(Psz, sizeof(double), N, f) != N ||
        fread(Ex,  sizeof(double), N, f) != N ||
        fread(Ey,  sizeof(double), N, f) != N ||
        fread(Ez,  sizeof(double), N, f) != N) {
        fprintf(stderr, "Checkpoint data read failed\n");
        goto cleanup;
    }

    CHECK(cudaMemcpy(Pmx_old_d, Pmx, Nvox * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Pmy_old_d, Pmy, Nvox * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Pmz_old_d, Pmz, Nvox * sizeof(double), cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(Psx_old_d, Psx, Nvox * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Psy_old_d, Psy, Nvox * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Psz_old_d, Psz, Nvox * sizeof(double), cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(Ex_d, Ex, Nvox * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Ey_d, Ey, Nvox * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Ez_d, Ez, Nvox * sizeof(double), cudaMemcpyHostToDevice));

    printf("Checkpoint %s loaded (peloop=%d, count=%d, Ez_ext=%g, incEz=%g)\n",
           fname, *peloop_out, *count_out, *Ez_ext_out, *incEz_out);

cleanup:
    if (f) fclose(f);

    free(Pmx);
    free(Pmy);
    free(Pmz);
    free(Psx);
    free(Psy);
    free(Psz);
    free(Ex);
    free(Ey);
    free(Ez);
}

void save_snapshot(const char *tag,
                   double *Pmx_d, double *Pmy_d, double *Pmz_d,
                   double *Psx_d, double *Psy_d, double *Psz_d, double P0_scale)
{
    char fnamePm[256], fnamePs[256];


    sprintf(fnamePm,  "output/microstructures/%s_pm.bin",  tag);
    sprintf(fnamePs,  "output/microstructures/%s_ps.bin",  tag);

    size_t nbytes = nx * ny * nz * sizeof(double);

    double *Pmx_h = (double*)malloc(nbytes);
    double *Pmy_h = (double*)malloc(nbytes);
    double *Pmz_h = (double*)malloc(nbytes);
    double *Psx_h = (double*)malloc(nbytes);
    double *Psy_h = (double*)malloc(nbytes);
    double *Psz_h = (double*)malloc(nbytes);

    CHECK(cudaMemcpy(Pmx_h, Pmx_d, nbytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(Pmy_h, Pmy_d, nbytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(Pmz_h, Pmz_d, nbytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(Psx_h, Psx_d, nbytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(Psy_h, Psy_d, nbytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(Psz_h, Psz_d, nbytes, cudaMemcpyDeviceToHost));

    // Scale by P0_scale 
    write_vector_field_binary(fnamePm, Pmx_h, Pmy_h, Pmz_h, nx, ny, nz, P0_scale);

    write_vector_field_binary(fnamePs, Psx_h, Psy_h, Psz_h, nx, ny, nz, P0_scale);

    free(Pmx_h);
    free(Pmy_h);
    free(Pmz_h);
    free(Psx_h);
    free(Psy_h);
    free(Psz_h);
}

