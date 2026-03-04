#ifndef GLOBALS_H
#define GLOBALS_H

#include "types.h"
#include <stdio.h>

#define PATHBUF 4000

extern MaterialParamsRaw FE_raw, AFE_raw, DE_raw;
extern MaterialNorm N_FE, N_AFE, N_DE;
extern MaterialNorm *grain_norm;

extern double maxEz, delEz;
extern int num_steps, count, time_to_change;
extern int initcount, device_flag, out_flag;
extern int dimx, dimy, dimz;
extern int nx, ny, nz, n_cout, ng_total, np;
extern int incr, ndiv, peloop;

extern double t0, l0, A;
extern double Eap, delEap, Ez_prev, Ez_ext, incEz;
extern double E0_ref, f0_ref, t0_ref;
extern double P0_ref, alpha_ref, L_ref, kappa_ref;

extern double del_h, del_t;
extern double sim_time, total_time;

extern FILE *fpout;
extern FILE *fvoln;

extern int offset, line_num;
extern int current_ref_type;
extern int max_sims;

extern int h_n_layers;
extern int h_layer_counts[16];
extern double afe_frac_layer[20], fe_frac_layer[20], de_frac_layer[20];

extern char crystal_type[20];
extern char z_bc_type[20];
extern int zbc_flag;

extern double eps_0_nd;
extern bool saved_Emax, saved_E0, saved_Emin;
extern int restarted_from_checkpoint;
extern int sim_index;

extern double one_by_nxnynz;

extern int n_samples;

extern double *alpha_m, *beta_m, *gamma_m, *g_m, *a0_m;
extern double *kappa_nd_m, *L_nd_m, *eps_m;
extern double *Ez_scale_m;

extern double *TR_h;

extern double *eta;
extern int    *omega;

extern double *phi, *theta, *psi;
extern double *centx, *centy, *centz;

extern double *inp_ga, *out_ga;

extern int *grain_ids, *grain_mat;
extern int *material_type;

// Poisson/CG solver parameters (used in depol_varEPS.cu)
extern const double g_bot;
extern const double g_top;
extern const double tol_abs;
extern const double tol_rel;
extern const int    max_iter;

// host buffers used in evolve.cu
extern double *Pmx, *Pmy, *Pmz;
extern double *Psx, *Psy, *Psz;

extern FILE *fvoln;          // you use fvoln in evolve.cu

extern int dep_flag;
extern int do_pre_relax;

extern double eps_ref;       // evolve.cu passes eps_ref into kernels
extern double del_h;         // you use del_h
extern double del_t;         // you use del_t

// CUB reduce temp storage + events used in evolve.cu
extern void   *t_storage;
extern size_t  t_storage_bytes;

extern double delEap;     // you have delEap in global.cu
extern double PsNoise;    // you have PsNoise in global.cu

extern double G0, omg;    // used in input parsing

extern int dep_flag;      // get_InputParams uses dep_flag
extern int do_pre_relax;  // get_InputParams uses do_pre_relax

extern int* d_is_interface;   // or static global

extern char output_dir[PATHBUF];

extern int n_struct;

#endif
