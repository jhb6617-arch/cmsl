#ifndef TYPES_H
#define TYPES_H

#include <stddef.h>
#include <stdbool.h>
#include <math.h>

#define PI acos(-1.0)
#define COMPERR 1.0e-6
#define NBINS 50

enum { AFE_TYPE = 0, FE_TYPE = 1, DE_TYPE = 2 };
enum Z_BC { DIRICHLET = 0, NEUMANN = 1 };

typedef struct MaterialParamsRaw {
    double alpha, beta, gamma1, g, a0;
    int transition_type;
    double L, kappa;
    double eps_r;
} MaterialParamsRaw;

typedef struct MaterialNorm {
    double alpha_nd, beta_nd, gamma_nd, g_nd, a0_nd;
    double p0, f0, t0, E0;
    double L_nd, kappa_nd;
    double eps_r;
} MaterialNorm;

typedef struct PhaseRefs {
    double alpha_ref;
    double P0_ref;
    double L_ref;
    double kappa_ref;
} PhaseRefs;

static inline double s2(double x){ return x*x; }
static inline double s4(double x){ double x2=x*x; return x2*x2; }
static inline double clamp_nonneg(double x){ return (x>0.0? x : 0.0); }

#endif
