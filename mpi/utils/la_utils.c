#include <stdlib.h>
#include <cblas.h>
#include <lapacke.h>

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

void mat_vec_column_mult (double* A, int rows, double* vec, int vecLen, double* output, int ldo) {
    for (int i = 0; i<rows; i++) {
        for (int j = 0; j<vecLen; j++) {
            if (j<rows) 
                output[ldo*i+j] = A[rows*i+j] * vec[j];
            else
                output[ldo*i+j] = 0.0;
        }
    }
}

void SVD_reconstruct_matrix(int s, int d, double *U, double *S, double *VT, double *M)
// TODO: Debug
{
    // Allocate memory for temporary matrices
    double *temp1 = (double *)malloc(s * d * sizeof(double));

    // Multiply U and S, store result in temp1
    mat_vec_column_mult(U, s, S, d, temp1);

    // Multiply the result by VT using BLAS
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, s, d, d, 1.0, temp1, s, VT, d, 0.0, M, s);

    free(temp1);
}