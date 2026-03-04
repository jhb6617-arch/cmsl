#include "header.h"     
#include <stdio.h>      
#include <string.h>     
#include <stdlib.h>     
#include <ctype.h>      


void processInputParams(const char *input_params_file, const char *output_file) {
    // Read InputParams dynamically
    FILE *fin = fopen(input_params_file, "r");
    if (fin == NULL) {
        printf("Error: Unable to open %s\n", input_params_file);
        return;
    }

char line[256];

while (fgets(line, sizeof(line), fin) != NULL) {
    if (strncmp(line, "device_flag", 11) == 0) {
        sscanf(line, "device_flag = %d", &device_flag);
    } else if (strncmp(line, "nx", 2) == 0) {
        sscanf(line, "nx = %d", &nx);
    } else if (strncmp(line, "ny", 2) == 0) {
        sscanf(line, "ny = %d", &ny);
    } else if (strncmp(line, "nz", 2) == 0) {
        sscanf(line, "nz = %d", &nz);
    } else if (strncmp(line, "dep_flag", 8) == 0) {
        sscanf(line, "dep_flag = %d", &dep_flag);
    } else if (strncmp(line, "del_h", 5) == 0) {
        sscanf(line, "del_h = %lf", &del_h);
    } else if (strncmp(line, "del_t", 5) == 0) {
        sscanf(line, "del_t = %le", &del_t);
    } else if (strncmp(line, "delEap", 6) == 0) {
        sscanf(line, "delEap = %le", &delEap);
    } else if (strncmp(line, "Eap", 3) == 0) {
        sscanf(line, "Eap = %le", &Eap);
    } else if (strncmp(line, "num_steps", 9) == 0) {
        sscanf(line, "num_steps = %d", &num_steps);
    } else if (strncmp(line, "out_flag", 8) == 0) {
        sscanf(line, "out_flag = %d", &out_flag);
    } else if (strncmp(line, "dimx", 4) == 0) {
        sscanf(line, "dimx = %d", &dimx);
    } else if (strncmp(line, "dimy", 4) == 0) {
        sscanf(line, "dimy = %d", &dimy);
    } else if (strncmp(line, "dimz", 4) == 0) {
        sscanf(line, "dimz = %d", &dimz);
    } else if (strncmp(line, "kappa_fe", 8) == 0) {
        sscanf(line, "kappa_fe = %le", &FE_raw.kappa);
    } else if (strncmp(line, "L_fe", 4) == 0) {
        sscanf(line, "L_fe = %lf", &FE_raw.L);
    } else if (strncmp(line, "kappa_afe", 9) == 0) {
        sscanf(line, "kappa_afe = %le", &AFE_raw.kappa);
    } else if (strncmp(line, "L_afe", 4) == 0) {
        sscanf(line, "L_afe = %lf", &AFE_raw.L);
    } else if (strncmp(line, "kappa_de", 8) == 0) {
        sscanf(line, "kappa_de = %le", &DE_raw.kappa);
    } else if (strncmp(line, "L_de", 4) == 0) {
        sscanf(line, "L_de = %lf", &DE_raw.L);
    } else if (strncmp(line, "PsNoise", 7) == 0) {
        sscanf(line, "PsNoise = %lf", &PsNoise);
    } else if (strncmp(line, "do_pre_relax", 12) == 0) {
        sscanf(line, "do_pre_relax = %d", &do_pre_relax);
    } else if (strncmp(line, "crystal_type", 12) == 0) {
        sscanf(line, "crystal_type = %s", crystal_type);
    } else if (strncmp(line, "G0", 2) == 0) {
        sscanf(line, "G0 = %lf", &G0);
    } else if (strncmp(line, "eps_de", 6) == 0) {
        sscanf(line, "eps_de = %lf", &DE_raw.eps_r);
    } else if (strncmp(line, "eps_fe", 6) == 0) {
        sscanf(line, "eps_fe = %lf", &FE_raw.eps_r);
    } else if (strncmp(line, "eps_afe", 7) == 0) {
        sscanf(line, "eps_afe = %lf", &AFE_raw.eps_r);
    } else if (strncmp(line, "omg", 3) == 0) {
        sscanf(line, "omg = %lf", &omg);
    }else if (strncmp(line, "z_bc_type", 9) == 0) {
        sscanf(line, "z_bc_type = %s", z_bc_type);
    } else if (strncmp(line, "n_layers", 8) == 0) {
        sscanf(line, "n_layers = %d", &h_n_layers);
    } else if (strncmp(line, "layer_counts", 12) == 0) {
        char* p = strchr(line, '=') + 1;
        int i = 0, offset = 0;
        while (i < h_n_layers && sscanf(p, "%d%n", &h_layer_counts[i], &offset) == 1) {
          p += offset;
          i++;
        }
        if (i != h_n_layers) {
           fprintf(stderr, "Error: expected %d layer_counts, got %d\n", h_n_layers, i);
           exit(1);
        }
    } 

}

fclose(fin);

    printf("\n--- Parameters Read ---\n");

      printf("device_flag = %d\n", device_flag);
      printf("nx = %d, ny = %d, nz = %d\n", nx, ny, nz);
      printf("del_h = %le, del_t = %le\n", del_h, del_t);
      printf("num_steps = %d\n", num_steps);
      printf("delEap = %le, Eap = %le\n", delEap, Eap);
      printf("dimx = %d, dimy = %d, dimz = %d\n", dimx, dimy, dimz);

      printf("Noise added to Ps = %le\n", PsNoise);

      printf("Depolarization flag = %s\n", dep_flag ? "Yes" : "No");
      printf("Using relaxation before applying field = %s\n", do_pre_relax ? "Yes" : "No");


    printf("-----------------------\n\n");

printf("  n_layers = %d\n", h_n_layers);
printf("  layer_counts = ");
for (int i = 0; i < h_n_layers; i++) {
    printf("%d ", h_layer_counts[i]);
}
printf("\n");

printf("-------- Type of Simulation: %s crystal ---------\n", crystal_type);

int sum = 0;
for (int i=0;i<h_n_layers;i++) sum += h_layer_counts[i];
if (sum != nz) {
    fprintf(stderr, "Error: layer_counts (%d) != nz (%d)\n", sum, nz);
    exit(1);
}


FILE *fout = fopen(output_file, "w");
if (fout == NULL) {
    printf("Error: Unable to open %s for writing\n", output_file);
    return;
}

fprintf(fout, "Device Flag: %d\n\n", device_flag);
fprintf(fout, "Print Flag for Ps and Pm: %d\n\n", out_flag);
fprintf(fout, "Grid Spacing (del_h): %lf\n\n", del_h);
fprintf(fout, "Time Step (del_t): %lf\n\n", del_t);
fprintf(fout, "Electric Field Increment (delE): %lf\n\n", delEap);
fprintf(fout, "Maximum Electric Field (maxE): %lf\n\n", Eap);
fprintf(fout, "Grid Points (nx, ny, nz): %d, %d, %d\n\n", nx, ny, nz);
fprintf(fout, "Number of Steps: %d\n\n", num_steps);
fprintf(fout, "Grid Dimensions (dimx, dimy, dimz): %d, %d, %d\n\n", dimx, dimy, dimz);

fprintf(fout, "\nNoise added to Ps: %le\n\n", PsNoise);

fprintf(fout, "---------------------------------\n");

fclose(fout);
}
