#include <stdio.h>
#include <mpi.h>
#include <gsl/gsl_matrix.h>
#include "utils/jpeg_to_matrix.h"

gsl_matrix* PCA_img_compression(gsl_matrix *P){
    // TODO: implementation
    return P;
}

int main(int argc, char **argv) {
    char *input_filename;
    char *output_filename = "output.jpeg";

    // Ensure that the input filename is provided as a command-line argument
    if (argc != 2) {
        printf("Usage: %s <input_filename.jpg>\n", argv[0]);
        return 1;
    }
    input_filename = argv[1];

    // Read from JPEG to GSL matrix
    gsl_matrix *img = read_JPEG_To_GSL(input_filename);

    img = PCA_img_compression(img);

    // Write from GSL matrix to JPEG
    write_GSL_to_JPEG(output_filename, img);

    // Free GSL matrix memory
    gsl_matrix_free(img);

    return 0;
}