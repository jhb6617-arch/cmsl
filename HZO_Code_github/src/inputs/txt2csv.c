#include <stdio.h>
#include <stdlib.h>

int main(void) {
    FILE *fin, *fout;
    double x, y;

    fin = fopen("../case2/sweep_0001.txt", "r");   // tab-delimited file
    if (fin == NULL) {
        perror("Error opening input file");
        return 1;
    }

    fout = fopen("sweep_0001.csv", "w"); // new CSV file
    if (fout == NULL) {
        perror("Error opening output file");
        fclose(fin);
        return 1;
    }

    while (fscanf(fin, "%le%le", &x, &y) == 2) {
        fprintf(fout, "%le,%le\n", x, y);
    }

    fclose(fin);
    fclose(fout);

    printf("Conversion done! Check output.csv\n");
    return 0;
}
