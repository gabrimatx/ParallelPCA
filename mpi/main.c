#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <lapacke.h>
#include <cblas.h>
#include "utils/jpeg_to_matrix.h"

// Function to print a matrix
void print_matrix(char *name, int rows, int cols, double *A, int lda)
{
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f\t", A[i * lda + j]);
        }
        printf("\n");
    }
    printf("\n");
}

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

void *decenter_dataset(int s, int d, double *M, double *mean)
{
    // Add the mean
    for (int i = 0; i < s; i++)
    {
        cblas_daxpy(d, 1, mean, 1, M + (i * d), 1);
    }
}

void SVD(int s, int d, double *M, double *U, double *S, double *VT)
{
    int lda = d;
    int ldu = s;
    int ldvt = d;
    int info;

    // Perform SVD
    info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', s, d, M, lda, S, U, ldu, VT, ldvt);
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
    int t;

    // Read input image
    if (my_rank == 0)
    {
        char *input_filename;

        // Ensure that the input filename is provided as a command-line argument
        if (argc != 3)
        {
            printf("Usage: %s <input_filename.jpg> <n_components>\n", argv[0]);
            return 1;
        }
        input_filename = argv[1];
        t = atoi(argv[2]);

        // Read from JPEG to matrix
        int s;
        img = read_JPEG_to_matrix(input_filename, &s, &d);
        local_s = s / comm_sz;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Broadcast local matrix dimensions and number of components
    MPI_Bcast(&local_s, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&t, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter local matrices to nodes
    double *local_img = (double *)malloc(sizeof(double) * local_s * d);

    MPI_Scatter(img, local_s * d, MPI_DOUBLE,
                local_img, local_s * d, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Center the dataset
    double *mean = center_dataset(local_s, d, local_img);

    // Perform SVD
    double *U_local = (double *)malloc(local_s * local_s * sizeof(double));
    double *D_local = (double *)malloc(d * sizeof(double));
    double *E_localT = (double *)malloc(d * d * sizeof(double));
    SVD(local_s, d, local_img, U_local, D_local, E_localT);

    // Set singular values after the t-th one to 0
    cblas_dscal(d - t, 0.0, D_local + t, 1);

    // Compute Pt_local

    // Compute St_local

    // Compute St with reduce

    // Do eigendecomposition of St

    // Obtain Pp_local by projecting Pt_local on Et (first t columns of E)
    // Dimensionalities: (Pp_local, local_s x d), (Pt_local, local_s x d), (Et, d x t), (Et.T, t x d)
    double *Pp_local  = (double *)malloc(local_s * d * sizeof(double));
    double *Pt_localEt = (double *)malloc(local_s * t * sizeof(double)); 
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, local_s, t, d, 1, Pt_local, local_s, Et, d, 0, Pt_localEt, local_s);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, local_s, d, t, 1, Pt_localEt, local_s, Et, d, 0, Pp_local, local_s);

    // Add the mean back to Pp_local
    center_dataset(local_s, d, Pp_local);

    // Gather all the Pp_local in Pp
    double *Pp;
    if (my_rank == 0)
    {
        Pp = (double *)malloc(sizeof(double) * s * d);
    }
    MPI_Gather(local_img, local_s * d, MPI_DOUBLE, Pp, s * d, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Output matrices to JPEG (to debug)
    char output_filename[20];
    sprintf(output_filename, "output%d.jpeg", my_rank);
    write_matrix_to_JPEG(output_filename, U_local, local_s, local_s);

    // Output Pp to JPEG
    write_matrix_to_JPEG("output.jpeg", Pp, s, d);

    // Free space and finalize
    if (my_rank == 0)
    {
        free(img);
    }
    free(U_local);
    free(D_local);
    free(E_localT);
    free(local_img);

    MPI_Finalize();

    return 0;
}