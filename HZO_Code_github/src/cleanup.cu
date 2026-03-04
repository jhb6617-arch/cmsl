#include "header.h"
#include "check.h"


void Cleanup()
{
    // Polarization fields
    CHECK(cudaFree(Psx_d));
    CHECK(cudaFree(Psy_d));
    CHECK(cudaFree(Psz_d));

    CHECK(cudaFree(Pmx_d));
    CHECK(cudaFree(Pmy_d));
    CHECK(cudaFree(Pmz_d));

    CHECK(cudaFree(Psx_old_d));
    CHECK(cudaFree(Psy_old_d));
    CHECK(cudaFree(Psz_old_d));

    CHECK(cudaFree(Pmx_old_d));
    CHECK(cudaFree(Pmy_old_d));
    CHECK(cudaFree(Pmz_old_d));

    // Microstructure
    CHECK(cudaFree(eta_d));
    CHECK(cudaFree(phi_d));
    CHECK(cudaFree(psi_d));
    CHECK(cudaFree(omega_d));

    // Material parameter arrays
    CHECK(cudaFree(alpha_m));
    CHECK(cudaFree(beta_m));
    CHECK(cudaFree(gamma_m));
    CHECK(cudaFree(g_m));
    CHECK(cudaFree(eps_m));
    CHECK(cudaFree(L_nd_m));
    CHECK(cudaFree(kappa_nd_m));
    CHECK(cudaFree(a0_m));

    CHECK(cudaFree(phi_old_d));

    CHECK(cudaFree (delecX_d));
    CHECK(cudaFree (delecY_d));
    CHECK(cudaFree (delecZ_d));
    CHECK(cudaFree (divP_d));
    
    // Grain-related
    CHECK(cudaFree(material_type));
    CHECK(cudaFree(grain_ids));
    CHECK(cudaFree(grain_norm));

    CHECK_CUDA(cudaFree(phi_dep_d));
    CHECK_CUDA(cudaFree(Pmx_local_d));
    CHECK_CUDA(cudaFree(Psx_local_d));
    CHECK_CUDA(cudaFree(Pmy_local_d));
    CHECK_CUDA(cudaFree(Psy_local_d));
    CHECK_CUDA(cudaFree(Pmz_local_d));
    CHECK_CUDA(cudaFree(Psz_local_d));
    CHECK_CUDA(cudaFree(d_is_interface));

    CHECK_CUDA(cudaFree(d_grain_mat));

    free (inp_ga);
    free (out_ga);
    free (eta);
    free (omega);
    free (phi);
    free (theta);
    free (psi);
    free (centx);
    free (centy);
    free (centz);
    free (grain_mat);

    cudaDeviceSynchronize();
}
