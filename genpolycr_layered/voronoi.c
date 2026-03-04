#include "parameters.h"

static inline int uniq_pick(int *used, int used_n, int candidate) {
    for (int i = 0; i < used_n; ++i) if (used[i] == candidate) return 0;
    return 1;
}

int main(void)
{
    FILE *fout;
    char fnchr[128];
    int i, j, h, k;

    // ----------------- alloc -----------------
    eta    = (double *)malloc(sizeof(double) * nx * ny * nz);
    omega  = (int    *)malloc(sizeof(int)    * nx * ny * nz);
    phi    = (double *)malloc(sizeof(double) * nor);
    theta  = (double *)malloc(sizeof(double) * nor);
    psi    = (double *)malloc(sizeof(double) * nor);

    angle  = (int    *)malloc(sizeof(int) * np);
    Aphi   = (double *)malloc(sizeof(double) * np);
    Atheta = (double *)malloc(sizeof(double) * np);
    Apsi   = (double *)malloc(sizeof(double) * np);

    double *centx = (double*)malloc(sizeof(double) * np);
    double *centy = (double*)malloc(sizeof(double) * np);
    double *centz = (double*)malloc(sizeof(double) * np);

    if (!eta || !omega || !phi || !theta || !psi || !angle || !Aphi || !Atheta || !Apsi || !centx || !centy || !centz) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    // ----------------- init -----------------
    memset(eta,   0, sizeof(double) * (size_t)nx*ny*nz);
    for (size_t t = 0; t < (size_t)nx*ny*nz; ++t) omega[t] = -1;

    for (i = 0; i < nor; ++i) { phi[i]=0.0; theta[i]=0.0; psi[i]=0.0; }

    FILE *fin = fopen("inputs/IODrandom_40000.DAT", "r");
    if (!fin) { fprintf(stderr, "Cannot open inputs/IODrandom_40000.DAT\n"); return 1; }
    int idx;
    double gphi, gtheta, gpsi;
    for (i = 0; i < nor; ++i) {
        if (fscanf(fin, "%d %lf %lf %lf", &idx, &gphi, &gtheta, &gpsi) != 4) break;
        if (idx >= 1 && idx <= nor) {
            phi   [idx-1] = gphi   * PI / 180.0;
            theta [idx-1] = gtheta * PI / 180.0;
            psi   [idx-1] = gpsi   * PI / 180.0;
        }
    }
    fclose(fin);

    // ----------------- seed rng -----------------
    srand((unsigned)time(NULL));

    // ----------------- randomize grain size -----------------
    // ng_total controls grain count: more grains = smaller size, fewer = larger
    // Must stay below np (max centroid storage = 2000)
    ng_total = 400 + rand() % 1401;   // random in [400, 1800]

    // ----------------- set per-layer grain counts (proportional) -----------------
    int sumz = 0;
    for (int L = 0; L < n_layers; ++L) sumz += layer_counts[L];

    int acc = 0;
    for (int L = 0; L < n_layers; ++L) {
        ng_layer[L] = (int) llround((double)layer_counts[L] / (double)sumz * (double)ng_total);
        acc += ng_layer[L];
    }
    // fix rounding to match exactly ng_total
    while (acc < ng_total) { ng_layer[0]++; acc++; }
    while (acc > ng_total) { ng_layer[0]--; acc--; }

    // quick sanity
    if (np < ng_total) {
        fprintf(stderr, "np (%d) must be >= ng_total (%d)\n", np, ng_total);
        return 1;
    }

    // ----------------- mkdir output dir -----------------
    int count = 0;
    sprintf(fnchr, "mkdir -p polc%04d", count);
    system(fnchr);

    // ----------------- layer-by-layer tessellation -----------------
    int global_offset = 0;   // starting global grain ID for current layer
    int z0 = 0;              // starting z slice for current layer

    for (int L = 0; L < n_layers; ++L) {
        const int z_span = layer_counts[L];
        const int z1     = z0 + z_span;
        const int nL     = ng_layer[L];

        printf("Layer %d: z=[%d..%d), thickness=%d, grains=%d, global_offset=%d\n",
               L, z0, z1, z_span, nL, global_offset);

        // ---- unique random orientations for this layer (nL of them) ----
        int used_n = 0;
        while (used_n < nL) {
            int pick = rand() % nor;
            if (uniq_pick(angle, used_n, pick)) angle[used_n++] = pick;
        }
        for (k = 0; k < nL; ++k) {
            int a = angle[k];
            Aphi  [global_offset + k] = phi  [a];
            Atheta[global_offset + k] = theta[a];
            Apsi  [global_offset + k] = psi  [a];
        }

        // ---- centroids for this layer (absolute coordinates) ----
        for (k = 0; k < nL; ++k) {
            centx[global_offset + k] = (double)(rand() % nx);
            centy[global_offset + k] = (double)(rand() % ny);
            centz[global_offset + k] = (double)(z0 + (rand() % z_span));
        }

        // ---- Voronoi tessellation restricted to this layer's z-slab ----
        // Periodic in x,y with periods nx,ny. Periodic in z WITHIN the slab (period = z_span).
        double rmin, r2;
        int nmin;
        for (i = 0; i < nx; ++i) {
            for (j = 0; j < ny; ++j) {
                for (h = z0; h < z1; ++h) {
                    rmin = 1e300;
                    nmin = 0;
                    for (k = 0; k < nL; ++k) {
                        const double cx = centx[global_offset + k];
                        const double cy = centy[global_offset + k];
                        const double cz = centz[global_offset + k];

                        // minimal image in x
                        double dx = cx - (double)i;
                        if      (dx >  0.5*nx) dx -= nx;
                        else if (dx < -0.5*nx) dx += nx;
                        // minimal image in y
                        double dy = cy - (double)j;
                        if      (dy >  0.5*ny) dy -= ny;
                        else if (dy < -0.5*ny) dy += ny;
                        // minimal image in z WITHIN LAYER (period = z_span)
                        // shift to local coordinates relative to slab origin z0
                        double hz = (double)h - (double)z0;
                        double cz_local = cz - (double)z0;
                        double dz = cz_local - hz;
                        if      (dz >  0.5*z_span) dz -= z_span;
                        else if (dz < -0.5*z_span) dz += z_span;

                        r2 = dx*dx + dy*dy + dz*dz;
                        if (r2 < rmin) { rmin = r2; nmin = k; }
                    }

                    const long idx3 = h + (j + i * ny) * nz;  // z-fastest
                    eta  [idx3] = 1.0;
                    omega[idx3] = global_offset + nmin;       // global unique ID
                }
            }
        }

        // advance to next layer
        global_offset += nL;
        z0 = z1;
    }

    // ----------------- outputs -----------------
    // total grains = ng_total (== sum ng_layer)
    sprintf(fnchr, "polc%04d/gnumb.txt", count);
    fout = fopen(fnchr, "w");
    fprintf(fout, "%d\n", ng_total);
    fclose(fout);

    // VTK
    sprintf(fnchr,"polc%04d/gdata.vtk", count);
    fout = fopen(fnchr, "w");
    fprintf(fout, "# vtk DataFile Version 3.0\n");
    fprintf(fout, "polycrystal\n");
    fprintf(fout, "ASCII\n");
    fprintf(fout, "DATASET STRUCTURED_GRID\n");
    fprintf(fout, "DIMENSIONS %d %d %d\n", nx, ny, nz);
    fprintf(fout, "POINTS %d double\n", nx*ny*nz);
    for (h = 0; h < nz; ++h)
        for (j = 0; j < ny; ++j)
            for (i = 0; i < nx; ++i)
                fprintf(fout, "%lf %lf %lf ", (double)i, (double)j, (double)h);

    fprintf(fout, "\nPOINT_DATA %d\n", nx*ny*nz);
    fprintf(fout, "SCALARS Grain_PF double\n");
    fprintf(fout, "LOOKUP_TABLE default\n");
    for (h = 0; h < nz; ++h)
        for (j = 0; j < ny; ++j)
            for (i = 0; i < nx; ++i)
                fprintf(fout, "%lf ", eta[h + (j + i * ny) * nz]);

    fprintf(fout, "\nSCALARS Grain_ID int\n");
    fprintf(fout, "LOOKUP_TABLE default\n");
    for (h = 0; h < nz; ++h)
        for (j = 0; j < ny; ++j)
            for (i = 0; i < nx; ++i)
                fprintf(fout, "%d ", omega[h + (j + i * ny) * nz]);
    fclose(fout);

    // Binary dump (omega, eta)
    sprintf(fnchr, "polc%04d/gdata.bin", count);
    fout = fopen(fnchr, "wb");
    fwrite(omega, sizeof(int),    (size_t)nx*ny*nz, fout);
    fwrite(eta,   sizeof(double), (size_t)nx*ny*nz, fout);
    fclose(fout);

    // Grain characteristics (centroids + orientations), one line per global grain ID
    // We generated centroids/orientations per layer at indices [global_offset..).
    sprintf(fnchr, "polc%04d/gchar.txt", count);
    fout = fopen(fnchr, "w");
    int written = 0, base = 0, zstart = 0;
    for (int L = 0; L < n_layers; ++L) {
        int nL = ng_layer[L];
        for (k = 0; k < nL; ++k) {
            int gid = base + k;
            fprintf(fout, "%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",
                    gid,
                    centx[gid], centy[gid], centz[gid],
                    Aphi[gid], Atheta[gid], Apsi[gid]);
            written++;
        }
        base += nL;
        zstart += layer_counts[L];
    }
    // (optional) assert written == ng_total
    fclose(fout);

    // ----------------- free -----------------
    free(centx); free(centy); free(centz);
    free(angle); free(Aphi);  free(Atheta); free(Apsi);
    free(eta);   free(omega);
    free(phi);   free(theta); free(psi);

    return 0;
}
