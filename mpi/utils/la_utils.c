#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <lapacke.h>
#include "io_utils.h"

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

void eigen_decomposition(int n, double *A, double *eigenvalues, double *eigenvectors)
{
    // LAPACK variables
    char jobz = 'V';       // 'V' means compute eigenvectors
    char uplo = 'U';       // 'U' means upper triangular part of A is used
    lapack_int lda = n;    // Leading dimension of A
    lapack_int lwork = -1; // Set to -1 to query optimal workspace size
    lapack_int info;

    // Workspace variables
    double wkopt;

    // Query and allocate workspace
    LAPACK_dsyev(&jobz, &uplo, &n, A, &lda, eigenvalues, &wkopt, &lwork, &info);
    lwork = (lapack_int)wkopt;
    double *work = (double *)malloc(lwork * sizeof(double));

    // Actual eigendecomposition
    LAPACK_dsyev(&jobz, &uplo, &n, A, &lda, eigenvalues, work, &lwork, &info);

    // Copy eigenvectors to the output array
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            eigenvectors[i * n + j] = A[i * lda + j];
        }
    }

    // Free workspace
    free(work);
}

void mat_vec_column_mult(double *A, int rows, int cols, double *vec, int vec_len, double *output)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (j < vec_len)
                output[cols * i + j] = A[rows * i + j] * vec[j];
            else
                output[cols * i + j] = 0.0;
        }
    }
}

// Note: rows_A, rows_B, cols_A and cols_B are supposed after the eventual transposition.
void multiply_matrices(double *A, int rows_A, int cols_A, int transposeA,
                       double *B, int rows_B, int cols_B, int transposeB, double *result)
{
    // Check if multiplication is possible
    if (cols_A != rows_B)
    {
        fprintf(stderr, "Matrix multiplication not possible: Invalid dimensions.\n");
        return;
    }

    // Perform matrix multiplication
    for (int i = 0; i < rows_A; ++i)
    {
        for (int j = 0; j < cols_B; ++j)
        {
            result[i * cols_B + j] = 0; // Initialize to zero before accumulating values
            for (int k = 0; k < cols_A; ++k)
            {
                int index_A = transposeA ? (k * cols_A + i) : (i * cols_A + k);
                int index_B = transposeB ? (j * cols_A + k) : (k * cols_B + j);
                result[i * cols_B + j] += A[index_A] * B[index_B];
            }
        }
    }
}

void SVD_reconstruct_matrix(int s, int d, double *U, double *S, double *VT, double *M)
{
    double *temp = (double *)malloc(s * d * sizeof(double));
    // Multiply U and S, store result in temp1
    mat_vec_column_mult(U, s, d, S, d, temp);

    // Multiply the result by VT using BLAS
    multiply_matrices(temp, s, d, 0, VT, d, d, 0, M);
}