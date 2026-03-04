#ifndef IO_UTILS_H
#define IO_UTILS_H

void write_field(const char *filename, int *Pmz, int nx, int ny, int nz, double p0);

void write_midplane_slice(const char *fname,
                          const double *Psz_host,
                          int nx, int ny, int nz);

void write_vector_field_binary(const char* filename,
                               double *Px, double *Py, double *Pz,
                               int nx, int ny, int nz, double p0);

void write_checkpoint(const char *fname,
                      double *Pmx_old_d, double *Pmy_old_d, double *Pmz_old_d,
                      double *Psx_old_d, double *Psy_old_d, double *Psz_old_d,
                      double *Ex_d, double *Ey_d, double *Ez_d,
                      int nx, int ny, int nz,
                      double Ez_ext, int peloop, int count, double incEz);

void read_checkpoint(const char *fname,
                     double *Pmx_old_d, double *Pmy_old_d, double *Pmz_old_d,
                     double *Psx_old_d, double *Psy_old_d, double *Psz_old_d,
                     double *Ex_d, double *Ey_d, double *Ez_d,
                     int nx, int ny, int nz,
                     int *peloop_out, int *count_out,
                     double *Ez_ext_out, double *incEz_out);

void save_snapshot(const char *tag,
                   double *Pmx_d, double *Pmy_d, double *Pmz_d,
                   double *Psx_d, double *Psy_d, double *Psz_d,
                   double P0_scale);

#endif
