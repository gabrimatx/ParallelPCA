#include <stdio.h>
#include <mpi.h>
#include <gsl/gsl_matrix.h>
#include "utils/jpeg_to_matrix.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int my_rank;
    int comm_sz;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    gsl_matrix *img;
    int local_s;
    int d;
    
    if (my_rank == 0) {
        char *input_filename;

        // Ensure that the input filename is provided as a command-line argument
        if (argc != 2) {
            printf("Usage: %s <input_filename.jpg>\n", argv[0]);
            return 1;
        }
        input_filename = argv[1];

        // Read from JPEG to GSL matrix
        img = read_JPEG_To_GSL(input_filename);
        local_s = img->size1 / comm_sz;
        d = img->size2;
    }
    MPI_Bcast(&local_s, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_INT, 0, MPI_COMM_WORLD);

    gsl_matrix *local_img = gsl_matrix_alloc(local_s, d);

    MPI_Scatter(img->data, local_s * d, MPI_DOUBLE,
                local_img->data, local_s * d, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    if (my_rank == 0) {
        gsl_matrix_free(img);
    }

    // TODO: Round 1
    

    // TODO: Round 2

    MPI_Finalize();

    return 0;
}