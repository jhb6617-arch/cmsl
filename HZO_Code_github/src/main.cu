#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#include "header.h"
#include "check.h"

__global__ void build_interface_flag(const int* omega,
    const int* grain_mat, int* is_interface, int nx, int ny, int nz)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny * nz;
    if (id >= N) return;

    int gid = omega[id];
    if (gid < 0) { is_interface[id] = 0; return; }

    int mat = grain_mat[gid];
    if (mat != 1) { is_interface[id] = 0; return; } // only FE voxels

    int k = id / (nx * ny);
    if (k == 0 || k == nz - 1) {
        is_interface[id] = 0;
        return;
    }

    int id_up = id + nx * ny;
    int id_dn = id - nx * ny;

    int gid_up = omega[id_up];
    int gid_dn = omega[id_dn];

    int mat_up = (gid_up >= 0) ? grain_mat[gid_up] : -1;
    int mat_dn = (gid_dn >= 0) ? grain_mat[gid_dn] : -1;

    is_interface[id] = (mat_up != 1 || mat_dn != 1);
}

void BuildInterfaceFlag()
{
    size_t Nvox = (size_t)nx * ny * nz;
    int t = 256;
    int b = (int)((Nvox + t - 1) / t);

    build_interface_flag<<<b, t>>>(omega_d, d_grain_mat, d_is_interface, nx, ny, nz);
    CHECK_KERNEL();
}


static void trim_token_inplace(char *s) {
    s[strcspn(s, " \t\r\n")] = '\0';
}

static int ref_type_from_str(const char *s) {
    if (!strcmp(s, "AFE")) return 0;
    if (!strcmp(s, "FE"))  return 1;
    if (!strcmp(s, "DE"))  return 2;
    return -1;
}

static void reset_evolve_state(void) {
    peloop = 0;          count = 0;            sim_time = 0.0;
    Ez_ext = 0.0;        Ez_prev = 0.0;        incEz = 0.0;
    saved_Emax = saved_E0 = saved_Emin = false;
}

// =========================================================
// Phase fraction pool (separated from Landau coefficients)
// =========================================================
#define MAX_PHASE_POOL 20000
static double g_afe_pool[MAX_PHASE_POOL];
static double g_fe_pool[MAX_PHASE_POOL];
static double g_de_pool[MAX_PHASE_POOL];
static int    g_pool_size = 0;

static void LoadPhasePool(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("Error opening phase_pool.txt"); exit(1); }

    g_pool_size = 0;
    while (g_pool_size < MAX_PHASE_POOL) {
        double a, f, d;
        if (fscanf(fp, "%lf %lf %lf", &a, &f, &d) != 3) break;
        if (d > 70.0) continue;   // DE fraction must be <= 70%
        g_afe_pool[g_pool_size] = a / 100.0;
        g_fe_pool[g_pool_size]  = f / 100.0;
        g_de_pool[g_pool_size]  = d / 100.0;
        g_pool_size++;
    }
    fclose(fp);
    printf("Phase pool loaded: %d entries (DE<=70%%) from '%s'\n", g_pool_size, filename);
}

static void PickPhaseFractions(double *afe_out, double *fe_out, double *de_out)
{
    int idx = rand() % g_pool_size;
    *afe_out = g_afe_pool[idx];
    *fe_out  = g_fe_pool[idx];
    *de_out  = g_de_pool[idx];
}

// =========================================================
// Landau coefficient pool (alpha,beta,gamma1,g,a0 per phase)
// L, kappa, eps_r stay fixed from InputParams
// =========================================================
typedef struct {
    double alpha, beta, gamma1, g, a0;  // AFE
    double fe_alpha, fe_beta, fe_gamma1, fe_g, fe_a0;  // FE
    double de_alpha, de_a0;             // DE
    int ref_type;
} LandauSet;

#define MAX_LANDAU_POOL 2000
static LandauSet g_landau_pool[MAX_LANDAU_POOL];
static int g_landau_pool_size = 0;

static void LoadLandauPool(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("Error opening landau pool file"); exit(1); }

    g_landau_pool_size = 0;
    char line[512];
    while (g_landau_pool_size < MAX_LANDAU_POOL) {
        if (!fgets(line, sizeof(line), fp)) break;
        char *p = line;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '#') continue;

        LandauSet ls;
        char ref_str[8];
        int n = sscanf(p,
            "%lf %lf %lf %lf %lf "
            "%lf %lf %lf %lf %lf "
            "%lf %lf %7s",
            &ls.alpha,    &ls.beta,    &ls.gamma1,    &ls.g,    &ls.a0,
            &ls.fe_alpha, &ls.fe_beta, &ls.fe_gamma1, &ls.fe_g, &ls.fe_a0,
            &ls.de_alpha, &ls.de_a0,  ref_str);
        if (n != 13) continue;

        trim_token_inplace(ref_str);
        ls.ref_type = ref_type_from_str(ref_str);
        if (ls.ref_type < 0) continue;

        g_landau_pool[g_landau_pool_size++] = ls;
    }
    fclose(fp);
    printf("Landau pool loaded: %d entries from '%s'\n", g_landau_pool_size, filename);
}

/* Copy one randomly picked Landau set into AFE_raw/FE_raw/DE_raw.
   L, kappa, eps_r are preserved (they come from InputParams). */
static void PickLandauSet(int *picked_idx_out)
{
    int idx = rand() % g_landau_pool_size;
    const LandauSet *ls = &g_landau_pool[idx];

    AFE_raw.alpha  = ls->alpha;    AFE_raw.beta  = ls->beta;
    AFE_raw.gamma1 = ls->gamma1;   AFE_raw.g     = ls->g;
    AFE_raw.a0     = ls->a0;

    FE_raw.alpha   = ls->fe_alpha; FE_raw.beta   = ls->fe_beta;
    FE_raw.gamma1  = ls->fe_gamma1;FE_raw.g      = ls->fe_g;
    FE_raw.a0      = ls->fe_a0;

    DE_raw.alpha   = ls->de_alpha; DE_raw.beta   = 0.0;
    DE_raw.gamma1  = 0.0;          DE_raw.g      = 0.0;
    DE_raw.a0      = ls->de_a0;

    current_ref_type = ls->ref_type;
    *picked_idx_out  = idx;
}

// =========================================================
// Parameter file parsing  (Landau only, 13 fields + label)
// =========================================================


static const MaterialParamsRaw* pick_ref_raw(int ref_type, const char **name_out)
{
    switch (ref_type) {
        case 0: *name_out = "AFE"; return &AFE_raw;
        case 1: *name_out = "FE";  return &FE_raw;
        case 2: *name_out = "DE";  return &DE_raw;
        default:
            *name_out = "FE";
            fprintf(stderr,
                    "Warning: current_ref_type=%d unknown, defaulting FE as reference.\n",
                    ref_type);
            return &FE_raw;
    }
}

double compute_p0(double alpha, double beta, double g, double gamma1, int transition_type)
{
    if (transition_type == 2) {
        return 2.0 * sqrt((g - alpha) / beta);
    } else {
        double disc = beta * beta - 4.0 * gamma1 * (alpha - g);
        if (disc < 0.0 && disc > -1e-14) disc = 0.0;
        if (disc < 0.0) return 1.0;

        double Num = 2.0 * fabs(beta) + 2.0 * sqrt(disc);
        if (gamma1 == 0.0) return 1.0;
        return sqrt(Num / gamma1);
    }
}

PhaseRefs make_phase_refs(MaterialParamsRaw raw)
{
    PhaseRefs R;
    R.alpha_ref = fabs(raw.alpha);

    if (raw.beta == 0.0 && raw.gamma1 == 0.0 && raw.g == 0.0) {
        R.P0_ref = 1.0;
    } else {
        R.P0_ref = compute_p0(raw.alpha, raw.beta, raw.g, raw.gamma1, raw.transition_type);
        if (!isfinite(R.P0_ref) || R.P0_ref <= 0.0) R.P0_ref = 1.0;
    }

    R.L_ref     = raw.L;
    R.kappa_ref = raw.kappa;
    return R;
}

MaterialNorm make_nd_wrt_reference(MaterialParamsRaw raw, const PhaseRefs Ref)
{
    MaterialNorm N;
    memset(&N, 0, sizeof(MaterialNorm));

    N.alpha_nd = raw.alpha / Ref.alpha_ref;

    if (!(raw.beta == 0.0 && raw.gamma1 == 0.0 && raw.g == 0.0)) {
        N.beta_nd  = (raw.beta   * Ref.P0_ref * Ref.P0_ref) / (8.0  * Ref.alpha_ref);
        N.gamma_nd = (raw.gamma1 * pow(Ref.P0_ref, 4.0))    / (48.0 * Ref.alpha_ref);
        N.g_nd     =  raw.g / Ref.alpha_ref;
    } else {
        N.beta_nd  = 0.0;
        N.gamma_nd = 0.0;
        N.g_nd     = 0.0;
    }

    N.a0_nd = (4.0 * raw.a0) / Ref.alpha_ref;

    N.L_nd     = 0.25 * raw.L * Ref.alpha_ref * t0;
    N.kappa_nd = raw.kappa / (Ref.alpha_ref * l0 * l0);

    N.eps_r = raw.eps_r;
    return N;
}

// -------------------------
// Setup + run one simulation
// -------------------------
void SetupAndRunSimulation(void)
{
    const char *ref_name = NULL;
    const MaterialParamsRaw *Ref_raw = pick_ref_raw(current_ref_type, &ref_name);

    PhaseRefs Rref = make_phase_refs(*Ref_raw);

    alpha_ref = Rref.alpha_ref;
    P0_ref    = Rref.P0_ref;
    L_ref     = Rref.L_ref;
    kappa_ref = Rref.kappa_ref;

    f0_ref = 0.25 * alpha_ref * P0_ref * P0_ref;
    E0_ref = f0_ref / P0_ref;

    printf("\n--- SetupAndRunSimulation ---\n");
    printf("Reference material: %s (current_ref_type=%d)\n", ref_name, current_ref_type);
    printf("alpha_ref = %e\n", alpha_ref);
    printf("P0_ref    = %e\n", P0_ref);
    printf("L_ref     = %e\n", L_ref);
    printf("kappa_ref = %e\n", kappa_ref);
    printf("f0_ref    = %e\n", f0_ref);
    printf("E0_ref    = %e\n", E0_ref);

    // Global effective fractions (for logging)
    double T_afe = 0.0, T_fe = 0.0, T_de = 0.0;
    double total_weight = 0.0;

    for (int L = 0; L < h_n_layers; ++L) {
        double wL = (double)h_layer_counts[L];

        double f_afe = afe_frac_layer[L];
        double f_fe  = fe_frac_layer[L];
        double f_de  = de_frac_layer[L];

        if (f_afe < 0.0) f_afe = 0.0;
        if (f_fe  < 0.0) f_fe  = 0.0;
        if (f_de  < 0.0) f_de  = 0.0;

        double S = f_afe + f_fe + f_de;
        if (S <= 0.0) { f_afe = 1.0; f_fe = 0.0; f_de = 0.0; S = 1.0; }

        f_afe /= S; f_fe /= S; f_de /= S;

        T_afe        += wL * f_afe;
        T_fe         += wL * f_fe;
        T_de         += wL * f_de;
        total_weight += wL;
    }

    double afe_frac_global = 0.0, fe_frac_global = 0.0, de_frac_global = 0.0;
    if (total_weight > 0.0) {
        afe_frac_global = T_afe / total_weight;
        fe_frac_global  = T_fe  / total_weight;
        de_frac_global  = T_de  / total_weight;
    }

    printf("Global effective fractions (0–1): AFE=%.3f, FE=%.3f, DE=%.3f\n",
           afe_frac_global, fe_frac_global, de_frac_global);

    PhaseRefs Rg;
    Rg.alpha_ref = alpha_ref;
    Rg.P0_ref    = P0_ref;
    Rg.L_ref     = L_ref;
    Rg.kappa_ref = kappa_ref;

    N_FE  = make_nd_wrt_reference(FE_raw,  Rg);
    N_AFE = make_nd_wrt_reference(AFE_raw, Rg);
    N_DE  = make_nd_wrt_reference(DE_raw,  Rg);

    printf("\n--- Nondimensional parameters (compact) ---\n");
    printf("AFE_nd: alpha=%+0.4f  beta=%+0.4f  gamma=%+0.4f  g=%+0.4f  a0=%+0.4f  kappa=%+0.4f  L=%+0.4f\n",
           N_AFE.alpha_nd, N_AFE.beta_nd, N_AFE.gamma_nd,
           N_AFE.g_nd, N_AFE.a0_nd, N_AFE.kappa_nd, N_AFE.L_nd);
    printf("FE_nd : alpha=%+0.4f  beta=%+0.4f  gamma=%+0.4f  g=%+0.4f  a0=%+0.4f  kappa=%+0.4f  L=%+0.4f\n",
           N_FE.alpha_nd, N_FE.beta_nd, N_FE.gamma_nd,
           N_FE.g_nd, N_FE.a0_nd, N_FE.kappa_nd, N_FE.L_nd);
    printf("DE_nd : alpha=%+0.4f  beta=%+0.4f  gamma=%+0.4f  g=%+0.4f  a0=%+0.4f  kappa=%+0.4f  L=%+0.4f\n",
           N_DE.alpha_nd, N_DE.beta_nd, N_DE.gamma_nd,
           N_DE.g_nd, N_DE.a0_nd, N_DE.kappa_nd, N_DE.L_nd);
    printf("------------------------------------------------------\n\n");

    double a1 = fabs(Ref_raw->alpha);
    eps_0_nd  = 4.0 / (8.854e-12 * a1);
    printf("Nondimensional relative permittvity eps_0_nd = %0.4f\n", eps_0_nd);

    maxEz = Eap    / E0_ref;
    delEz = delEap / E0_ref;
    printf("Nondimensional max field %0.4f and increment %0.4f\n", maxEz, delEz);

    Init_Conf();
    BuildInterfaceFlag();

    time_to_change = num_steps;
    initcount = 0;

    reset_evolve_state();
    Evolve();
}

int main(int argc, char *argv[])
{
    int rc = 0;

    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <offset_sims> <landau_pool_file> <phase_pool_file> [max_sims]\n"
            "Example: %s 0 inputs/landau_pool.txt inputs/phase_pool.txt 500\n",
            argv[0], argv[0]);
        return 1;
    }

    int offset_sims = atoi(argv[1]);
    if (offset_sims < 0) offset_sims = 0;

    const char *landau_file = argv[2];
    const char *phase_file  = argv[3];

    if (argc >= 5) {
        max_sims = atoi(argv[4]);
        if (max_sims < 1) max_sims = 1;
    } else {
        max_sims = 500;  // default: 500 sims per KISTI job
    }

    // Use offset_sims to seed differently across KISTI jobs
    srand((unsigned int)(time(NULL) + (unsigned int)offset_sims * 1234567u));

    // Fixed params (L, kappa, eps_r, grid dims, etc.)
    processInputParams("inputs/InputParams", "OutputParams");
    printf("Parameters processed successfully\n");
    printf(">>> offset_sims=%d, will run %d simulations\n", offset_sims, max_sims);

    // Load pools
    LoadLandauPool(landau_file);
    LoadPhasePool(phase_file);

    // grain tag
    char grain_tag[16];
    if      (!strcmp(crystal_type, "single")) strcpy(grain_tag, "single");
    else if (!strcmp(crystal_type, "poly"))   strcpy(grain_tag, "poly");
    else { fprintf(stderr, "Invalid crystal_type\n"); return 1; }

    // init GPU + data (allocate for max possible ng_total)
    one_by_nxnynz = 1.0 / ((double)nx * ny * nz);
    cudaSetDevice(device_flag);

    ng_total = 8500;
    AllocateData();
    const int ng_max = ng_total;  // hard upper limit for grain arrays

    evolve_resources_init();

    int total_size = nx * ny * nz;
    threadsPerBlock = 128;
    numBlocks = (total_size + threadsPerBlock - 1) / threadsPerBlock;

    threads = dim3(dimx, dimy, dimz);
    blocks  = dim3((nx + threads.x - 1) / threads.x,
                   (ny + threads.y - 1) / threads.y,
                   (nz + threads.z - 1) / threads.z);

    // z BC
    if      (!strcmp(z_bc_type, "diri")) zbc_flag = DIRICHLET;
    else if (!strcmp(z_bc_type, "neu"))  zbc_flag = NEUMANN;
    else { fprintf(stderr, "Unsupported z_bc_type: %s\n", z_bc_type); rc = 1; goto cleanup; }
    cudaMemcpyToSymbol(d_zbc_type, &zbc_flag, sizeof(int));

    // Auto-detect number of strucN folders in inputs/
    n_struct = 0;
    {
        struct stat _st;
        char _test[256];
        for (int s = 1; ; ++s) {
            snprintf(_test, sizeof(_test), "inputs/struc%d", s);
            if (stat(_test, &_st) != 0 || !S_ISDIR(_st.st_mode)) break;
            n_struct = s;
        }
    }
    if (n_struct == 0) { fprintf(stderr, "No inputs/struc1 folder found. Exiting.\n"); rc = 1; goto cleanup; }
    printf("Found %d structure(s) in inputs/\n", n_struct);

    // Single flat output directory: output/
    snprintf(output_dir, sizeof(output_dir), "output");
    make_dir_p(output_dir);
    printf("Output dir: %s\n", output_dir);

    // =========================================================
    // Main random-sampling loop  (500 sims per KISTI job)
    // Each sim independently picks:
    //   1 Landau set  (from 1000-set pool)
    //   1 Phase fraction set  (from 1000-set pool, DE<=70%)
    //   1 Polycrystalline structure  (from n_struct folders)
    // =========================================================
    for (int global_sim = 0; global_sim < max_sims; ++global_sim) {

        sim_index = offset_sims + global_sim + 1;
        printf("\n==== Simulation %d (job_sim %d/%d) ====\n",
               sim_index, global_sim + 1, max_sims);

        // --- 1. Pick random Landau coefficient set ---
        int landau_idx = -1;
        PickLandauSet(&landau_idx);
        printf("Landau set #%d  ref_type=%d\n", landau_idx, current_ref_type);

        // --- 2. Pick random phase fractions ---
        double picked_afe, picked_fe, picked_de;
        PickPhaseFractions(&picked_afe, &picked_fe, &picked_de);
        for (int L = 0; L < h_n_layers; L++) {
            afe_frac_layer[L] = picked_afe;
            fe_frac_layer[L]  = picked_fe;
            de_frac_layer[L]  = picked_de;
        }
        printf("Fractions: AFE=%.1f%%  FE=%.1f%%  DE=%.1f%%\n",
               picked_afe * 100.0, picked_fe * 100.0, picked_de * 100.0);

        // --- 3. Pick random polycrystalline structure ---
        int sid = 1 + (rand() % n_struct);
        char struct_dir[256];
        snprintf(struct_dir, sizeof(struct_dir), "inputs/struc%d", sid);
        printf("Structure: %s\n", struct_dir);

        ReadGrainCount(struct_dir, grain_tag);
        if (ng_total > ng_max) {
            fprintf(stderr,
                "Warning: ng_total=%d > ng_max=%d for %s, skipping.\n",
                ng_total, ng_max, struct_dir);
            ng_total = ng_max;
        }
        LoadGrainStructureFromDir(struct_dir, grain_tag);

        // Reset depolarization field so it reinitialises for new structure
        phi_dep_initialized = false;

        // --- 4. Run simulation ---
        reset_evolve_state();
        SetupAndRunSimulation();
    }

    printf("\n##### Finished %d simulations #####\n", max_sims);

cleanup:
    evolve_resources_cleanup();
    Cleanup();
    return rc;
}
