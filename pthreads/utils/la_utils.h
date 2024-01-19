#ifndef LA_UTILS_H
#define LA_UTILS_H

void dataset_partial_mean(int s, int local_s, int d, double *M, double *mean);
void center_dataset(int s, int d, double *M, double *mean);
void decenter_dataset(int s, int d, double *M, double *mean);
void SVD(int s, int d, double *M, double *U, double *S, double *VT);
void eigen_decomposition(int n, double *A, double *L);
void mat_vec_column_mult(double *A, int rows, double *vec, int vec_len, double *output);
void multiply_matrices(double *A, int rows_A, int cols_A, int transposeA, double *B, int rows_B, int cols_B, int transposeB, double *result);
void reverse_matrix_columns(double *A, int rows, int cols, int tda, double *At);
void SVD_reconstruct_matrix(int s, int d, double *U, double *S, double *VT, double *M);

#endif // LA_UTILS_H