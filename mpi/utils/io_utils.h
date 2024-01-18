#ifndef IO_UTILS_H
#define IO_UTILS_H

double *read_JPEG_to_matrix(char *filename, int *rows, int *cols);
void write_matrix_to_JPEG(char *filename, double *matrix, int rows, int cols);
void print_matrix(char *name, int rows, int cols, double *A, int lda);
void print_matrix_int(char *name, int rows, int cols, double *A, int lda);
void print_vector(char *name, int dim, double *v);
void auto_tester(char *filename, int rows, int cols, double *A, int lda);

#endif // IO_UTILS_H