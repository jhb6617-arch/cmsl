#include "../include/header.h"

MaterialParamsRaw FE_raw;
MaterialParamsRaw AFE_raw;
MaterialParamsRaw DE_raw;

MaterialNorm N_FE;
MaterialNorm N_AFE;
MaterialNorm N_DE;

MaterialNorm *grain_norm = NULL;

double *Pmx_old_d = NULL, *Pmy_old_d = NULL, *Pmz_old_d = NULL;
double *Pmx_d = NULL, *Pmy_d = NULL, *Pmz_d = NULL;

double *Psx_old_d = NULL, *Psy_old_d = NULL, *Psz_old_d = NULL;
double *Psx_d = NULL, *Psy_d = NULL, *Psz_d = NULL;

int num_steps = 0, count = 0, time_to_change = 0;
int initcount = 0, device_flag = 0, out_flag = 0;

int dimx = 0, dimy = 0, dimz = 0;

int nx = 0, ny = 0, nz = 0;
int n_cout = 0, n_samples = 153;
int ng_total = 0, np = 1000;

int incr = 0, ndiv = 0, peloop = 0;

double t0 = 3.0e-09; 
double l0 = 1.0e-09;  

double Eap = 0.0, delEap = 0.0;
double Ez_ext = 0.0;
double Ez_prev= 0.0;
double incEz = 0.0;   
double maxEz = 0.0, delEz = 0.0;
double sE_FE = 1.0, sE_AFE = 1.0, sE_DE = 1.0;
double sP_FE = 1.0, sP_AFE = 1.0, sP_DE = 1.0;
double E0_ref = 1.0, t0_ref = 1.0, A = 0.0, f0_ref = 1.0;

double P0_ref   = 1.0, alpha_ref = 1.0, L_ref = 1.0, kappa_ref = 1.0;

double eps_fe = 0.0, eps_de = 0.0, eps_afe = 0.0, eps_ref = 0.0;

double del_h = 0.0;
double del_t = 0.0;               
double sim_time = 0.0, total_time = 0.0; 

double Psum = 0.0, Pmean = 0.0;
double PsNoise = 0.0;

int sim_index = 1;

bool saved_Emax = false;
bool saved_E0   = false;
bool saved_Emin = false;

int *omega_d = NULL, *omega = NULL;
double *eta_d = NULL, *eta = NULL;
double *phi_d = NULL, *theta_d = NULL, *psi_d = NULL;
double *phi = NULL, *theta = NULL, *psi = NULL;

double *centx = NULL, *centy = NULL, *centz = NULL;

dim3 threads, blocks;
int threadsPerBlock = 0, numBlocks = 0;

int current_ref_type = 1;

int max_sims = INT_MAX;

int h_layer_counts[16] = {0};
int ng_layer[16] = {0};
int h_n_layers         = 0;
double afe_frac_layer[20] = {0}; 
double fe_frac_layer[20] = {0}; 
double de_frac_layer[20] = {0};

double *alpha_m = NULL, *beta_m = NULL, *gamma_m = NULL, *g_m = NULL, *a0_m = NULL;
double *kappa_nd_m = NULL;  // per-voxel κ_nd
double *L_nd_m = NULL;      // per-voxel ¼ * (L/L_ref
double *eps_m = NULL;      // per-voxel ¼ * (L/L_ref
double *Ez_scale_m = NULL;// *P_scale_m = NULL;

double *Pmx = NULL, *Pmy = NULL, *Pmz = NULL;
double *Psx = NULL, *Psy = NULL, *Psz = NULL;

double *inp_ga = NULL;
double *out_ga = NULL;

char crystal_type[20] = {0};

void *t_storage = NULL;
size_t t_storage_bytes = 0;
cudaEvent_t start, stop;

double *Psum_d = NULL;
char flnamePm[256];
char flnamePs[256];
FILE *fvoln = NULL;

FILE *fpout = NULL;

double *TR_d = NULL;
double *TR_h = NULL;

double L_nd = 0.0;
double kappa_nd = 0.0;

double de_frac = 0.0, afe_frac = 0.0, fe_frac = 0.0; 
int n_afe = 0, n_de = 0, n_fe = 0;
int *grain_ids = NULL, *material_type = NULL;

int offset = 0;

int line_num = 0;

double G0 = 0.0, omg = 0.0;

double *delecX_d = NULL;
double *delecY_d = NULL;
double *delecZ_d = NULL;
double *divP_d = NULL;
double *phi_old_d = NULL;

double *Pmx_local_d = NULL;
double *Psx_local_d = NULL;
double *Pmy_local_d = NULL;
double *Psy_local_d = NULL;
double *Pmz_local_d = NULL;
double *Psz_local_d = NULL;
int *d_grain_mat = NULL;
int *grain_mat = NULL;

int dep_flag = 0;

int do_pre_relax = 0;

const double g_bot   = 0.0;
const double g_top   = 0.0;
const double tol_abs     = 1.0e-3;
const double tol_rel     = 1.0e-2;
const int    max_iter= 4000;
double a1 = 0.0, eps_0_nd = 0.0;

double *phi_dep_d = NULL;
bool    phi_dep_initialized = false;
double one_by_nxnynz = 1.0;

int    restarted_from_checkpoint = 0;

int* d_is_interface;

__device__ __constant__ int d_zbc_type;
__device__ int d_ref_id = -1;

char z_bc_type[20] = {0};
int zbc_flag = 0;

char output_dir[PATHBUF] = "output";

int n_struct = 10;
