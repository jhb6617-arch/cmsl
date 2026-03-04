//#include "generateBin.c" 
#include "header.h"
#include "check.h"


__device__ __forceinline__ int IDX(int i,int j,int k,int nx,int ny){
    return i + nx * (j + k * ny);
}

__device__ __forceinline__ double harm(double a, double b){
    const double eps = 1e-20;
    double denom = a + b;
    if (denom < eps) denom = eps;
    return (2.0 * a * b) / denom;
}

// Fill an array with 1.0 (for computing averages with cublasDdot)
__global__ void fill_ones(double *arr, int N)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < N) arr[t] = 1.0;
}

// Add a uniform electric field vector to the E-field arrays
__global__ void add_uniform_E(double *Ex, double *Ey, double *Ez,
                              double Ex_uni, double Ey_uni, double Ez_uni,
                              int N)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;

    Ex[t] += Ex_uni;
    Ey[t] += Ey_uni;
    Ez[t] += Ez_uni;
}

__global__ void apply_A(double * __restrict__ y,
                        const double * __restrict__ x,
                        const double * __restrict__ eps_r,
                        int nx,int ny,int nz,double h,
                        double g_bot,double g_top)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    if (i>=nx || j>=ny || k>=nz) return;

    int id = IDX(i,j,k,nx,ny);

    const int bc = d_zbc_type;  // DIRICHLET or NEUMANN

    // --- Neumann gauge fix: pin one reference node so A is SPD ---
    if (bc == NEUMANN && id == d_ref_id) {
        // A*x = x at this node, RHS will be 0 -> phi(ref) = 0
        y[id] = x[id];
        return;
    }



    // --- Dirichlet planes: A x = x - g ---
    if (bc == DIRICHLET) {
        if (k == 0) {
            y[id] = x[id] - g_bot;
            return;
        }
        if (k == nz-1) {
            y[id] = x[id] - g_top;
            return;
        }
    }

    int ip=(i+1)%nx, im=(i-1+nx)%nx;
    int jp=(j+1)%ny, jm=(j-1+ny)%ny;

    double ec = eps_r[id];
    double ee = eps_r[IDX(ip,j ,k ,nx,ny)];
    double ew = eps_r[IDX(im,j ,k ,nx,ny)];
    double en = eps_r[IDX(i ,jp,k ,nx,ny)];
    double es = eps_r[IDX(i ,jm,k ,nx,ny)];

        double ex_p = harm(ec,ee), ex_m = harm(ec,ew);
    double ey_p = harm(ec,en), ey_m = harm(ec,es);

    // ------- Z direction (crucial fix) -------
    int up, dn;

    if (bc == DIRICHLET) {
        // simple one-sided for Dirichlet is OK
        up = k+1;
        dn = k-1;
    }
    else {  
        // -------- SECOND-ORDER NEUMANN --------
        // enforce: phi[-1] = phi[1], phi[n] = phi[n-2]
        if (k == 0)      { up = 1;   dn = 1; }
        else if (k == nz-1) { up = nz-2; dn = nz-2; }
        else             { up = k+1; dn = k-1; }
    }

    int idx_u = IDX(i,j,up,nx,ny);
    int idx_d = IDX(i,j,dn,nx,ny);

    double eu = eps_r[idx_u];
    double ed = eps_r[idx_d];

    double ez_p = harm(ec,eu);
    double ez_m = harm(ec,ed);

    double xc = x[id];
    double xe = x[IDX(ip,j ,k ,nx,ny)];
    double xw = x[IDX(im,j ,k ,nx,ny)];
    double xn = x[IDX(i ,jp,k ,nx,ny)];
    double xs = x[IDX(i ,jm,k ,nx,ny)];
    double xu = x[idx_u];
    double xd = x[idx_d];


    double Ax = ( ex_p*(xe - xc) - ex_m*(xc - xw)
                + ey_p*(xn - xc) - ey_m*(xc - xs)
                + ez_p*(xu - xc) - ez_m*(xc - xd) ) / (h*h);

    y[id] = -Ax;  // CG solves A x = b with A = -L
}

__global__ void build_diag_inv(double * __restrict__ dinv,
                               const double * __restrict__ eps_r,
                               int nx,int ny,int nz,double h)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    if (i>=nx || j>=ny || k>=nz) return;
    int id = IDX(i,j,k,nx,ny);

    const int bc = d_zbc_type;

    // Neumann gauge-fix node: A(row) = 1, so diag = 1, inverse = 1
    if (bc == NEUMANN && id == d_ref_id) {
        dinv[id] = 1.0;
        return;
    }

    if (bc == DIRICHLET && (k==0 || k==nz-1)) {
        dinv[id] = 1.0;
        return;
    }

    int ip=(i+1)%nx, im=(i-1+nx)%nx;
    int jp=(j+1)%ny, jm=(j-1+ny)%ny;

    double ec = eps_r[id];
    double ee = eps_r[IDX(ip,j ,k ,nx,ny)];
    double ew = eps_r[IDX(im,j ,k ,nx,ny)];
    double en = eps_r[IDX(i ,jp,k ,nx,ny)];
    double es = eps_r[IDX(i ,jm,k ,nx,ny)];

    double ex_p = harm(ec,ee), ex_m = harm(ec,ew);
    double ey_p = harm(ec,en), ey_m = harm(ec,es);

      int up,dn;
    if (bc==DIRICHLET) {
        up=k+1; dn=k-1;
    } else {
        // second-order Neumann
        if (k==0){ up=1; dn=1;}
        else if (k==nz-1){up=nz-2; dn=nz-2;}
        else { up=k+1; dn=k-1; }
    }

    double eu = eps_r[IDX(i,j,up,nx,ny)];
    double ed = eps_r[IDX(i,j,dn,nx,ny)];

    double ez_p = harm(ec,eu);
    double ez_m = harm(ec,ed);

    double D = (ex_p+ex_m + ey_p+ey_m + ez_p+ez_m)/(h*h);

    dinv[id] = (D>1e-30 ? 1.0/D : 1.0);
}

__global__ void applyJacobiPreconditioner(int N,
                                          const double * __restrict__ r,
                                          const double * __restrict__ dinv,
                                          double * __restrict__ z)
{
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= N) return;
    z[t] = r[t] * dinv[t];
}

__global__ void rhs_zero_on_planes(double *b, int nx,int ny,int nz)
{

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i>=nx || j>=ny) return;

    b[IDX(i,j,0,    nx,ny)] = 0.0;
    b[IDX(i,j,nz-1, nx,ny)] = 0.0;
}

__global__ void init_phi_linear(double* __restrict__ phi,
                                int nx,int ny,int nz,
                                double g_bot,double g_top)
{
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= (unsigned)nx || j >= (unsigned)ny || k >= (unsigned)nz) return;

    int id = IDX((int)i,(int)j,(int)k,nx,ny);

    if (d_zbc_type == DIRICHLET) {
        double t = (nz <= 1) ? 0.0 : (double)((int)k) / (double)(nz-1);
        double val = (1.0 - t) * g_bot + t * g_top;
        phi[id] = val;
    } else { // NEUMANN: simple constant initial guess
        phi[id] = 0.0;
    }
}

__global__ void Calc_EfieldMag(double *Ex, double *Ey, double *Ez, const double *phi,
                               double del_h, int nx, int ny, int nz,
                    	       double g_bot, double g_top, int count, double *divP_d)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int id = i + nx * (j + k * ny);

    // --- Periodic neighbors in x and y ---
    int e = (i + 1) % nx;
    int w = (i - 1 + nx) % nx;
    int n = (j + 1) % ny;
    int s = (j - 1 + ny) % ny;

    int idx_e = e + nx * (j + k * ny);
    int idx_w = w + nx * (j + k * ny);
    int idx_n = i + nx * (n + k * ny);
    int idx_s = i + nx * (s + k * ny);

    Ex[id] = -(phi[idx_e] - phi[idx_w]) / (2.0 * del_h);

    Ey[id] = -(phi[idx_n] - phi[idx_s]) / (2.0 * del_h);

 const int bc = d_zbc_type;

    int up,dn;

    if (bc==DIRICHLET) {
        up = (k==nz-1 ? k : k+1);
        dn = (k==0    ? k : k-1);

        double phi_up = (k==nz-1 ? g_top : phi[IDX(i,j,up,nx,ny)]);
        double phi_dn = (k==0    ? g_bot : phi[IDX(i,j,dn,nx,ny)]);

        Ez[id] = -(phi_up-phi_dn)/(2*del_h);
    } 
    else {  
        // 2nd-order Neumann
        if (k==0)      { up=1; dn=1; }
        else if (k==nz-1){ up=nz-2; dn=nz-2; }
        else           { up=k+1; dn=k-1; }

        double phi_up = phi[IDX(i,j,up,nx,ny)];
        double phi_dn = phi[IDX(i,j,dn,nx,ny)];

        Ez[id] = -(phi_up-phi_dn)/(2*del_h);
    }


}


__global__ void divergence(double *divP_d,
                           double *Pmx_d, double *Pmy_d, double *Pmz_d,
                           double del_h, int nx, int ny, int nz,
                           int count, double eps_0_nd)
{
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= nx || j >= ny || k >= nz) return;

    unsigned int id = i + nx * (j + k * ny);

    // --- Periodic boundaries in x- and y-direction ---
    int e = (i + 1) % nx;
    int w = (i - 1 + nx) % nx;
    int n = (j + 1) % ny;
    int s = (j - 1 + ny) % ny;

    // --- Neumann boundary in z-direction (consistent with TDGL) ---
    int u = (k == nz - 1) ? k : (k + 1);     // up neighbor index
    int d = (k == 0)      ? k : (k - 1);     // down neighbor index

    unsigned int idx_e = e + nx * (j + k * ny);
    unsigned int idx_w = w + nx * (j + k * ny);
    unsigned int idx_n = i + nx * (n + k * ny);
    unsigned int idx_s = i + nx * (s + k * ny);
    unsigned int idx_u = i + nx * (j + u * ny);
    unsigned int idx_d = i + nx * (j + d * ny);

    // --- Finite differences ---
    double Pm_x_e = Pmx_d[idx_e];
    double Pm_x_w = Pmx_d[idx_w];

    double Pm_y_n = Pmy_d[idx_n];
    double Pm_y_s = Pmy_d[idx_s];

    // Consistent Neumann in z: use clamped neighbors
    double Pm_z_f = Pmz_d[idx_u];   // forward
    double Pm_z_b = Pmz_d[idx_d];   // backward

    // --- Divergence: central difference (with boundary consistency) ---
    divP_d[id] = ( (Pm_x_e - Pm_x_w) +
                   (Pm_y_n - Pm_y_s) +
                   (Pm_z_f - Pm_z_b) ) / (2.0 * del_h);

    divP_d[id] *= eps_0_nd;
  //if (count > 100 && id == 5000)
  //   printf("divergence = %le\tcount = %d\n",  divP_d[id], count);

}


void Depol_field()
{
    /* ---- constants / sizes ---- */
    const int    N       = nx * ny * nz;
    const double delh    = del_h;

    /* ---- choose reference node for Neumann gauge fix ---- */
    if (zbc_flag == NEUMANN) {
        // pick an interior node (center-ish), away from z=0 and z=nz-1
        int i_ref = nx / 2;
        int j_ref = ny / 2;
        int k_ref = nz / 2;  // nz>=3 in your runs, so this is interior

        int h_ref_id = i_ref + nx * (j_ref + k_ref * ny);
        CHECK_CUDA(cudaMemcpyToSymbol(d_ref_id, &h_ref_id, sizeof(int)));
    }


    //------ One-time allocation & initialization of phi_dep_d ----
    if (!phi_dep_initialized) {
        CHECK_CUDA(cudaMalloc(&phi_dep_d, (size_t)N * sizeof(double)));

        // use same blocks/threads as the rest of depol solver
        init_phi_linear<<<blocks, threads>>>(phi_dep_d, nx, ny, nz, g_bot, g_top);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        phi_dep_initialized = true;
    }

    double *d_ones = NULL;
   
    /* ---- PCG work buffers ---- */
    double *d_x     = NULL;  /* solution phi */
    double *d_r     = NULL;  /* residual */
    double *d_z     = NULL;  /* preconditioned residual */
    double *d_p     = NULL;  /* search direction */
    double *d_Ap    = NULL;  /* A*p */
    double *d_dinv  = NULL;  /* Jacobi diag inverse */
    double *d_b     = NULL;  /* RHS = (1/eps0)*divP (modified on Dirichlet planes) */

    CHECK_CUDA(cudaMalloc(&d_x,     (size_t)N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_r,     (size_t)N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_z,     (size_t)N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_p,     (size_t)N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_Ap,    (size_t)N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_dinv,  (size_t)N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_b,     (size_t)N * sizeof(double)));

    /* ---- 1) divergence RHS: divP_d = (1/eps0)*div P ---- */
    divergence<<<blocks, threads>>>(divP_d, Pmx_d, Pmy_d, Pmz_d,
                                    delh, nx, ny, nz, count, eps_0_nd);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_KERNEL();

    /* cuBLAS setup */
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    const double one = 1.0;
    const double minus1 = -1.0;

    CHECK_CUDA(cudaMalloc(&d_ones, (size_t)N * sizeof(double)));
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    fill_ones<<<numBlocks, blockSize>>>(d_ones, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    /* d_b <- divP_d */
    CHECK_CUDA(cudaMemcpy(d_b, divP_d, (size_t)N*sizeof(double), cudaMemcpyDeviceToDevice));
    CHECK_CUBLAS(cublasDscal(cublasHandle, N, &minus1, d_b, 1));  // b = -b

    // ---- Neumann gauge-fix: RHS at pinned node must be 0 ----
    if (zbc_flag == NEUMANN) {
        int i_ref = nx / 2;
        int j_ref = ny / 2;
        int k_ref = nz / 2;
        int h_ref_id = i_ref + nx * (j_ref + k_ref * ny);

        double zero = 0.0;
        CHECK_CUDA(cudaMemcpy(d_b + h_ref_id, &zero, sizeof(double),
                              cudaMemcpyHostToDevice));
    }

    /* Zero RHS on Dirichlet planes (so Ax = x - g there with b=0) */
    if (zbc_flag == DIRICHLET) {
        dim3 th2d(16,16,1);
        dim3 bl2d((nx + th2d.x - 1)/th2d.x,
                  (ny + th2d.y - 1)/th2d.y, 1);
        rhs_zero_on_planes<<<bl2d, th2d>>>(d_b, nx, ny, nz);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    /* ---- 2) PCG solve for phi ---- */

    /* init guess: reuse phi from previous TDGL step */
    CHECK_CUDA(cudaMemcpy(d_x, phi_dep_d, (size_t)N * sizeof(double),
                      cudaMemcpyDeviceToDevice));

    /* build Jacobi preconditioner diagonal inverse from eps_m */
    build_diag_inv<<<blocks, threads>>>(d_dinv, eps_m, nx, ny, nz, delh);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    /* r = b - A*x */
    apply_A<<<blocks, threads>>>(d_Ap, d_x, eps_m, nx, ny, nz, delh, g_bot, g_top);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    /* r = b */
    CHECK_CUBLAS(cublasDcopy(cublasHandle, N, d_b, 1, d_r, 1));
    /* r = r - A*x */
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &minus1, d_Ap, 1, d_r, 1));

    /* z = M^{-1} r */
    {
        applyJacobiPreconditioner<<<numBlocks, blockSize>>>(N, d_r, d_dinv, d_z);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    /* p = z */
    CHECK_CUBLAS(cublasDcopy(cublasHandle, N, d_z, 1, d_p, 1));

    /* rdotz_old = r.z */
double rdotz_old = 0.0, rdotz_new = 0.0;
CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_r, 1, d_z, 1, &rdotz_old));
double rdotz0 = rdotz_old;

int    kiter   = 0;
bool   pcg_ok  = true;

// ---- initial residual sanity check ----
if (!isfinite(rdotz_old) || rdotz_old < 0.0) {
    printf("PCG ERROR: invalid initial residual rdotz_old = %g\n", rdotz_old);
    pcg_ok = false;   // skip iterations
} else {

    // ---- PCG loop with abs + rel tolerance ----
    while (kiter < max_iter &&
           rdotz_old > tol_abs * tol_abs &&
           (rdotz0 <= 0.0 || (rdotz_old / rdotz0) > tol_rel * tol_rel))
    {
        ++kiter;

        /* Ap = A*p */
        apply_A<<<blocks, threads>>>(d_Ap, d_p, eps_m, nx, ny, nz, delh, g_bot, g_top);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        /* denom = p^T Ap */
        double denom = 0.0;
        CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_p, 1, d_Ap, 1, &denom));
        if (!isfinite(denom) || !(denom > 0.0)) {
            printf("PCG breakdown: denom=%g at iter %d\n", denom, kiter);
            pcg_ok = false;
            break;
        }

        double alpha = rdotz_old / denom;

        /* x = x + alpha*p */
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1));

        /* r = r - alpha*Ap */
        double nalpha = -alpha;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &nalpha, d_Ap, 1, d_r, 1));

        /* z = M^{-1} r */
        applyJacobiPreconditioner<<<numBlocks, blockSize>>>(N, d_r, d_dinv, d_z);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        /* rdotz_new = r . z */
        CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_r, 1, d_z, 1, &rdotz_new));
        if (!isfinite(rdotz_new) || rdotz_new < 0.0) {
            printf("PCG ERROR: invalid residual rdotz_new = %g at iter %d\n",
                   rdotz_new, kiter);
            pcg_ok = false;
            break;
        }

        double beta = rdotz_new / rdotz_old;

        /* p = z + beta*p */
        CHECK_CUBLAS(cublasDscal(cublasHandle, N, &beta, d_p, 1));
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &one,  d_z, 1, d_p, 1));

        rdotz_old = rdotz_new;
    }
}

/*
// ---- diagnostic residual norms ----
double rdotz_final = 0.0;
double res0_norm   = 0.0;
double res_norm    = 0.0;

if (pcg_ok) {
    CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_r, 1, d_z, 1, &rdotz_final));
    res0_norm = sqrt(fabs(rdotz0));
    res_norm  = sqrt(fabs(rdotz_final));

    printf("PCG converged in %d iterations. res = %e, res0 = %e\n",
           kiter, res_norm, res0_norm);
} else {
    printf("PCG aborted at iter %d (see error above).\n", kiter);
}
*/
    // Save current phi for next TDGL step (warm start)
    CHECK_CUDA(cudaMemcpy(phi_dep_d, d_x, (size_t)N * sizeof(double), cudaMemcpyDeviceToDevice));


    /* ---- 3) Compute heterogeneous E field from phi ---- */
    Calc_EfieldMag<<<blocks, threads>>>(delecX_d, delecY_d, delecZ_d, d_x,
                                        delh, nx, ny, nz, g_bot, g_top, count, divP_d);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

/*
    if (zbc_flag == NEUMANN) {
        // Allocate and fill ones vector (for averaging)
        fill_ones<<<numBlocks, blockSize>>>(d_ones, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // Compute spatial averages of P (on device) using cublasDdot
        double  Pz_sum = 0.0;
        //CHECK_CUBLAS(cublasDdot(cublasHandle, N, Pmx_d, 1, d_ones, 1, &Px_sum));
        //CHECK_CUBLAS(cublasDdot(cublasHandle, N, Pmy_d, 1, d_ones, 1, &Py_sum));
        CHECK_CUBLAS(cublasDdot(cublasHandle, N, Pmz_d, 1, d_ones, 1, &Pz_sum));

        //double Px_avg = Px_sum * invN;
        //double Py_avg = Py_sum * invN;
        double Pz_avg = Pz_sum * one_by_nxnynz;

        // Uniform depolarization field (nondimensional form)
        // E_dep_uni = - (1/eps0)*P_avg  --> here we use eps_0_nd consistently
        double Ex_uni = 0.0;//-eps_0_nd * Px_avg;
        double Ey_uni = 0.0;//-eps_0_nd * Py_avg;
        double Ez_uni = -eps_0_nd * Pz_avg;

        // Add E_dep_uni to heterogeneous field from Poisson
        add_uniform_E<<<numBlocks, blockSize>>>(delecX_d, delecY_d, delecZ_d,
                                                Ex_uni, Ey_uni, Ez_uni, N);
       CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

    }

*/
    /* ---- cleanup ---- */
    cublasDestroy(cublasHandle);
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_r));
    CHECK_CUDA(cudaFree(d_z));
    CHECK_CUDA(cudaFree(d_p));
    CHECK_CUDA(cudaFree(d_Ap));
    CHECK_CUDA(cudaFree(d_dinv));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_ones));

}
