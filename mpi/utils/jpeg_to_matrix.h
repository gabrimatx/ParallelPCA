#ifndef JPEG_TO_MATRIX_H
#define JPEG_TO_MATRIX_H

#include <gsl/gsl_matrix.h>

gsl_matrix* read_JPEG_To_GSL(char* filename);
void write_GSL_to_JPEG(char* filename, gsl_matrix* matrix);

#endif  // JPEG_TO_MATRIX_H