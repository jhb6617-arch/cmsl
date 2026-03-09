#ifndef HEADER_H
#define HEADER_H

#include "types.h"
#include "globals.h"
#include "io_utils.h"

// only include cuda_deps.h inside .cu files:
#ifdef __CUDACC__
#include "cuda_deps.h"
#endif

#define PATHBUF 4000

// prototypes
double compute_p0(double alpha, double beta, double g, double gamma1, int transition_type);
PhaseRefs make_phase_refs(MaterialParamsRaw raw);
MaterialNorm make_nd_wrt_reference(MaterialParamsRaw raw, PhaseRefs Ref);

void processInputParams(const char *input_params_file, const char *output_file);
void ReadGrainCount(const char *dir, const char *grain_tag);
void LoadGrainStructureFromDir(const char *dir, const char *grain_tag);
void AllocateData(void);
void evolve_resources_init(void);
void evolve_resources_cleanup(void);
void Init_Conf(void);
void Evolve(void);
void SetupAndRunSimulation(void);
void Cleanup(void);

void Depol_field(void);

void write_checkpoint(const char *fname,
                      const double *Pmx, const double *Pmy, const double *Pmz,
                      const double *Psx, const double *Psy, const double *Psz,
                      const double *Ex,  const double *Ey,  const double *Ez,
                      int nx, int ny, int nz, double Ez_ext, int peloop, int count, double incEz);

#endif
