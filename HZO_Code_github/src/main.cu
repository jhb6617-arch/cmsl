#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>
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
    // cut at first whitespace or CR/LF
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

int readParamsFromFile(FILE *fp,
                       double *afe_frac_pct, double *fe_frac_pct,
                       double *de_frac_pct, int *ref_type_out)
{
    double afe_alpha, afe_beta, afe_gamma1, afe_g, afe_a0;
    double fe_alpha,  fe_beta,  fe_gamma1,  fe_g,  fe_a0;
    double de_alpha,  de_a0;
    double afe_frac,  fe_frac,  de_frac;
    char   ref_str[8];
    char   line[512];

    for (;;) {
        if (!fgets(line, sizeof(line), fp)) return 0; // EOF / read error

        // skip whitespace
        char *p = line;
        while (isspace((unsigned char)*p)) p++;

        // skip blank / comment
        if (*p == '\0' || *p == '\n' || *p == '#') continue;

        int n = sscanf(p, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
	                  "%lf %lf %lf %lf %lf %7s",
            &afe_alpha, &afe_beta, &afe_gamma1, &afe_g, &afe_a0,
            &fe_alpha,  &fe_beta,  &fe_gamma1,  &fe_g,  &fe_a0,
            &de_alpha,  &de_a0,  &afe_frac,  &fe_frac,  &de_frac, ref_str);

        if (n != 16) {
            fprintf(stderr,
                "readParamsFromFile: expected 16 fields, got %d in line:\n%s\n",
                n, line);
            return -1;
        }

        trim_token_inplace(ref_str);

        int ref_type = ref_type_from_str(ref_str);
        if (ref_type < 0) {
            fprintf(stderr, "readParamsFromFile: unknown Ref_mat '%s'\n", ref_str);
            return -1;
        }

        // --- Fill AFE raw ---
        AFE_raw.alpha  = afe_alpha;
        AFE_raw.beta   = afe_beta;
        AFE_raw.gamma1 = afe_gamma1;
        AFE_raw.g      = afe_g;
        AFE_raw.a0     = afe_a0;
        AFE_raw.transition_type = AFE_TYPE;

        // --- Fill FE raw ---
        FE_raw.alpha  = fe_alpha;
        FE_raw.beta   = fe_beta;
        FE_raw.gamma1 = fe_gamma1;
        FE_raw.g      = fe_g;
        FE_raw.a0     = fe_a0;
        FE_raw.transition_type = FE_TYPE;

        // --- Fill DE raw ---
        DE_raw.alpha  = de_alpha;
        DE_raw.beta   = 0.0;
        DE_raw.gamma1 = 0.0;
        DE_raw.g      = 0.0;
        DE_raw.a0     = de_a0;
        DE_raw.transition_type = DE_TYPE;

        // --- Fractions in % from file ---
        *afe_frac_pct = afe_frac;
        *fe_frac_pct  = fe_frac;
        *de_frac_pct  = de_frac;

        *ref_type_out = ref_type;
        return 1;
    }
}

typedef struct {
    double afe;     // 0..1
    double fe;      // 0..1
    double de;      // 0..1
    int    ref_type;
} LayerData;

static int read_one_layer(FILE *fp, LayerData *out, int *line_num)
{
    double afe_pct, fe_pct, de_pct;
    int ref_type;

    int status = readParamsFromFile(fp, &afe_pct, &fe_pct, &de_pct, &ref_type);
    if (status <= 0) return status; // 0 EOF, -1 parse error

    (*line_num)++; // counts records (layers)

    out->afe = afe_pct / 100.0;
    out->fe  = fe_pct  / 100.0;
    out->de  = de_pct  / 100.0;
    out->ref_type = ref_type;

    return 1;
}

// returns: 1 ok, 0 clean EOF at boundary, -2 unexpected EOF mid-block, -1 parse error
static int read_one_sim_block(FILE *fp, int n_layers, int *line_num, int *first_ref_type_out)
{
    int first_ref = -1;

    for (int L = 0; L < n_layers; ++L) {
        LayerData ld;
        int st = read_one_layer(fp, &ld, line_num);

        if (st == 0) return (L == 0) ? 0 : -2; // boundary vs mid-block EOF
        if (st < 0)  return -1;

        afe_frac_layer[L] = ld.afe;
        fe_frac_layer[L]  = ld.fe;
        de_frac_layer[L]  = ld.de;

        if (L == 0) first_ref = ld.ref_type;
        else if (ld.ref_type != first_ref)
            printf("Warning: ref_mat differs between layers. Using first layer value.\n");

        printf("Line %d (layer %d): AFE=%.1f%% FE=%.1f%% DE=%.1f%% Ref=%d\n",
               *line_num, L, ld.afe * 100.0, ld.fe * 100.0, ld.de * 100.0, ld.ref_type);
    }

    *first_ref_type_out = first_ref;
    return 1;
}

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
        // (your DE/linear special case — keeping your current behavior)
        return 2.0 * sqrt((g - alpha) / beta);
    } else {
        double disc = beta * beta - 4.0 * gamma1 * (alpha - g);
        if (disc < 0.0 && disc > -1e-14) disc = 0.0; // tiny numerical protection
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

    // Linear dielectric / nonpolar
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

// NOTE: this is NOT "local per-phase" because you pass the chosen reference scales (Rg).
// Rename it to match what it actually does:
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

    // kinetics / gradient (keeping your formulas)
    N.L_nd     = 0.25 * raw.L * Ref.alpha_ref * t0;
    N.kappa_nd = raw.kappa / (Ref.alpha_ref * l0 * l0);

    N.eps_r = raw.eps_r;
    return N;
}

// -------------------------
// Setup + run one simulation (one block)
// -------------------------

void SetupAndRunSimulation(void)
{
    // A) Reference material
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

    // B) Global effective fractions (for logging)
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

    // C) Build nondimensional parameters w.r.t chosen reference
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
    FILE *fp = NULL;

    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <offset_sims> <filename> [max_sims]\n"
            "Example: %s 0 phase_table.txt 5\n",
            argv[0], argv[0]);
        return 1;
    }

    // args
    int offset_sims = atoi(argv[1]);
    if (offset_sims < 0) offset_sims = 0;

    char filename[256];
    strncpy(filename, argv[2], sizeof(filename) - 1);
    filename[sizeof(filename) - 1] = '\0';

    // max_sims default comes from global.cu: int max_sims = INT_MAX;
    if (argc >= 4) {
        max_sims = atoi(argv[3]);
        if (max_sims < 1) max_sims = 1;
    }

    // params
    processInputParams("inputs/InputParams", "OutputParams");
    printf("Parameters processed successfully\n");

    offset = offset_sims * h_n_layers;
    printf("Skipping %d simulations (%d layers)\n", offset_sims, offset);
    printf(">>> Will run up to %d simulations from this offset\n", max_sims);

    // grain tag
    char grain_tag[16];
    if      (!strcmp(crystal_type, "single")) strcpy(grain_tag, "single");
    else if (!strcmp(crystal_type, "poly"))   strcpy(grain_tag, "poly");
    else { fprintf(stderr, "Invalid crystal_type\n"); return 1; }

    // init GPU + data
    one_by_nxnynz = 1.0 / ((double)nx * ny * nz);
    cudaSetDevice(device_flag);

    // Allocate with maximum possible grain count (voronoi generates up to 1800,
    // capped at np=2000). Per-structure ng_total is read inside the loop.
    ng_total = 2000;
    AllocateData();

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



for (int sid = 1; sid <= n_struct; ++sid) {

    // 1) set structure input folder
    char struct_dir[256];
    snprintf(struct_dir, sizeof(struct_dir), "inputs/struc%d", sid);

    // 2) set structure output folder (NO sim folder)
    // output/struc001, output/struc002, ...
    set_output_dir_for_structure(sid);

    // 3) open phase table fresh for this structure
    fp = fopen(filename, "r");
    if (!fp) { perror("Error opening input file"); rc = 1; goto cleanup; }

    // 4) skip offset rows (same as your old logic)
    line_num = 0;
    for (int s = 0; s < offset; ++s) {
        LayerData tmp;
        int st = read_one_layer(fp, &tmp, &line_num);
        if (st == 0) { printf("EOF while skipping offset.\n"); fclose(fp); fp = NULL; goto next_structure; }
        if (st < 0)  { printf("Parse error while skipping offset.\n"); rc = 1; goto cleanup; }
    }

    // Read this structure's grain count, then load its grain structure
    ReadGrainCount(struct_dir, grain_tag);
    LoadGrainStructureFromDir(struct_dir, grain_tag);

    // 5) now run ALL parameter sets for THIS structure
    for (long sims_started = 0; sims_started < max_sims; ++sims_started) {

        sim_index = offset_sims + (int)sims_started + 1;  // 1,2,3...

        printf("\n==== Structure %d, Param-set %d ====\n", sid, sim_index);

        int first_ref_type = -1;
        int st = read_one_sim_block(fp, h_n_layers, &line_num, &first_ref_type);

        if (st == 0) { printf("Reached EOF cleanly.\n"); break; }
        if (st == -2) { fprintf(stderr, "Unexpected EOF inside %d-layer block.\n", h_n_layers); rc = 1; goto cleanup; }
        if (st < 0)  { fprintf(stderr, "Parse error near record %d\n", line_num); rc = 1; goto cleanup; }

        current_ref_type = first_ref_type;

        // run this parameter-set
        reset_evolve_state();
        SetupAndRunSimulation();   // writes: output_dir/sweep001.txt, sweep002.txt, ...
    }

    fclose(fp);
    fp = NULL;

next_structure:
    if (fp) { fclose(fp); fp = NULL; }

    printf("##### Finished ALL parameter sets for structure %d #####\n", sid);
}


cleanup:
    if (fp) fclose(fp);
    evolve_resources_cleanup();
    Cleanup();
    return rc;
}

