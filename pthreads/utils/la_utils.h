#ifndef LA_UTILS_H
#define LA_UTILS_H

double *center_dataset(int s, int d, double *M);
void *decenter_dataset(int s, int d, double *M, double *mean);
void SVD(int s, int d, double *M, double *U, double *S, double *VT);
void eigen_decomposition(int n, double *A, double *eigenvalues, double *eigenvectors);
void mat_vec_column_mult(double *A, int rows, double *vec, int vec_len, double *output);
void multiply_matrices(double *A, int rows_A, int cols_A, double *B, int rows_B, int cols_B, double *result);
void SVD_reconstruct_matrix(int s, int d, double *U, double *S, double *VT, double *M);

#endif // LA_UTILS_H