#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define PI acos(-1.0)

int *omega;
double *eta;
double *phi, *theta, *psi;
double gphi, gtheta, gpsi;
double *Aphi, *Atheta, *Apsi;
int *angle;

// Thin film dimensions
int nx = 940, ny = 940, nz = 10, nx_half, ny_half, nz_half; // nz = layer_counts
long SEED = -9630614;
int np = 9000;      // max centroid storage (ng_total up to 8500)
int nor = 39999;    // orientation database size
double gsize;

// ------------------ Layering ------------------
int layer_counts[]  = {10};   // thickness of each layer in z , thickness : 10
int n_layers        = 1; // fixed for single layer

// ------------------ Grains ------------------
int ng_total = 800;        // total grains across the whole film , keep it
int ng_layer[15];          // grains per layer, filled at runtime
