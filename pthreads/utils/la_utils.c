#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <lapacke.h>
#include "io_utils.h"

void dataset_partial_mean(int s, int local_s, int d, double *M, double *mean)
{
    // Calculate the partial mean
    for (int j = 0; j < local_s; j++)
    {
        cblas_daxpy(d, 1.0 / s, M + (j * d), 1, mean, 1);
    }
}

void center_dataset(int s, int d, double *M, double *mean)
{
    // Subtract the mean
    for (int i = 0; i < s; i++)
    {
        cblas_daxpy(d, -1, mean, 1, M + (i * d), 1);
    }
}

void decenter_dataset(int s, int d, double *M, double *mean)
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

void eigen_decomposition(int n, double *A, double *L)
{
    int lda = n;
    LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', n, A, lda, L);
}

void mat_vec_column_mult(double *A, int rows, int cols, double *vec, int vec_len, double *output, int tdo)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (j < vec_len)
                output[tdo * i + j] = A[rows * i + j] * vec[j];
            else break;
        }
    }
}

// Note: rows_A, rows_B, cols_A and cols_B are supposed after the eventual transposition.
void multiply_matrices(double *A, int rows_A, int cols_A, int transposeA,
                       double *B, int rows_B, int cols_B, int transposeB,
                       double *result, int overwrite)
{
    // Check if multiplication is possible
    if (cols_A != rows_B)
    {
        fprintf(stderr, "Matrix multiplication not possible: Invalid dimensions.\n");
        return;
    }

    // Perform matrix multiplication
    for (int i = 0; i < rows_A; i++)
    {
        for (int j = 0; j < cols_B; j++)
        {
            if (overwrite)
            {
                result[i * cols_B + j] = 0; // Initialize to zero
            }
            for (int k = 0; k < cols_A; k++)
            {
                int index_A = transposeA ? (k * rows_A + i) : (i * cols_A + k);
                int index_B = transposeB ? (j * cols_A + k) : (k * cols_B + j);
                result[i * cols_B + j] += A[index_A] * B[index_B];
            }
        }
    }
}

void reverse_matrix_columns(double *A, int rows, int cols, int tda, double *At)
{
    double temp;
    int stop = (tda / 2) < cols ? (tda / 2) : tda - cols;
    for (int i = 0; i < rows; i++)
    {
        for (int j = tda - 1; j >= stop; j--)
        {
            temp = A[(i + 1) * tda - j - 1];
            At[i * cols + tda - j - 1] = A[i * tda + j];
            if (tda - j - 1 >= tda - cols)
            {
                At[i * cols + j] = temp;
            }
        }
    }
}

void SVD_reconstruct_matrix(int s, int d, double *U, double *S, double *VT, double *M)
{
    double *temp = (double *)calloc(s * d, sizeof(double));
    // Multiply U and S, store result in temp1
    mat_vec_column_mult(U, s, s, S, d, temp, d);

    // Multiply the result by VT
    multiply_matrices(temp, s, d, 0, VT, d, d, 0, M, 1);
    free(temp);
}

void accumulate_matrix (double *A, int rows, int cols, double* accumulation) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            accumulation[i*cols + j] += A[i*cols + j];
        }
    }
}


void set_local_extremes(double* A, int rows, int cols, double local_min, double local_max) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (local_min > A[i*cols + j]) A[i*cols + j] = local_min;
            if (local_max < A[i*cols + j]) A[i*cols + j] = local_max;
        }
    }
}

void get_local_extremes(double* A, int rows, int cols, double *local_min, double* local_max) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (*local_min > A[i*cols + j]) *local_min = A[i*cols + j];
            if (*local_max < A[i*cols + j]) *local_max = A[i*cols + j];
        }
    }
}

void rescale_image(double* img, int rows, int cols, double global_min, double global_max) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            img[i*cols + j] = 255.99*(img[i*cols + j] - global_min)/(global_max-global_min);
        }
    }
}
