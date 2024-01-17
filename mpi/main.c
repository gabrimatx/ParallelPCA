#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <lapacke.h>
#include <cblas.h>
#include "utils/jpeg_to_matrix.h"

double *center_dataset(int s, int d, double *M)
{
    // Initialize mean
    double *mean = (double *)malloc(sizeof(double) * d);
    for (int i = 0; i < d; i++)
    {
        mean[i] = 0.0;
    }

    // Calculate the mean
    for (int j = 0; j < s; j++)
    {
        cblas_daxpy(d, 1.0 / s, M + (j * d), 1, mean, 1);
    }

    // Subtract the mean
    for (int k = 0; k < s; k++)
    {
        cblas_daxpy(d, -1, mean, 1, M + (k * d), 1);
    }

    return mean;
}

double *decenter_dataset(int s, int d, double *M, double *mean)
{
    // Add the mean
    for (int i = 0; i < s; i++)
    {
        cblas_daxpy(d, 1, mean, 1, M + (i * d), 1);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int my_rank;
    int comm_sz;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    double *img;
    int local_s;
    int d;

    if (my_rank == 0)
    {
        char *input_filename;

        // Ensure that the input filename is provided as a command-line argument
        if (argc != 2)
        {
            printf("Usage: %s <input_filename.jpg>\n", argv[0]);
            return 1;
        }
        input_filename = argv[1];

        // Read from JPEG to GSL matrix
        int s;
        img = read_JPEG_to_matrix(input_filename, &s, &d);
        local_s = s / comm_sz;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&local_s, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double *local_img = (double *)malloc(sizeof(double) * local_s * d);

    MPI_Scatter(img, local_s * d, MPI_DOUBLE,
                local_img, local_s * d, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    char output_filename[20];

    // Center the dataset
    double *mean = center_dataset(local_s, d, local_img);
    decenter_dataset(local_s, d, local_img, mean);

    sprintf(output_filename, "output%d.jpeg", my_rank);
    write_matrix_to_JPEG(output_filename, local_img, local_s, d);

    // TODO: Round 1

    // TODO: Round 2

    if (my_rank == 0)
    {
        free(img);
    }

    MPI_Finalize();

    return 0;
}