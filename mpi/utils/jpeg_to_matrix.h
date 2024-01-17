#ifndef JPEG_TO_MATRIX_H
#define JPEG_TO_MATRIX_H

double* read_JPEG_to_matrix(char* filename, int* rows, int* cols);
void write_matrix_to_JPEG(char* filename, double* matrix, int rows, int cols);

#endif  // JPEG_TO_MATRIX_H