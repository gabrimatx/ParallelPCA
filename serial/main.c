#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include <cblas.h>
#include "utils/io_utils.h"
#include "utils/la_utils.h"

const double DBL_MIN = -1e5;
const double DBL_MAX = 1e5;
double global_min = 0.0, global_max = 255.99;

int main(int argc, char *argv[])
{

	// Ensure that the input filename is provided as a command-line argument
	char *input_filename;
	if (argc != 3 && argc != 4)
	{
		printf("Usage: %s <input_filename.jpg> <n_principal_components> <style (optional)>\n", argv[0]);
		return 1;
	}
	input_filename = argv[1];
	double *img;
	int s, d, t;
	img = read_JPEG_to_matrix(input_filename, &s, &d);
	t = atoi(argv[2]);
	if (t > d) {printf("ERROR: the number of Principal Components (%d) cannot be greater than the number of columns of the image (%d).\n\n", t, d); return 1;}
	
	int style = 0;
	if (argc == 4) style = atoi(argv[3]);

	// Allocate space needed
	double *mean = (double *)calloc(d, sizeof(double));
	double *St = (double *)calloc(d * d, sizeof(double));
	double *Et = (double *)malloc(d * t * sizeof(double));
	double *U = (double *)malloc(s * s * sizeof(double));
	double *D = (double *)malloc(d * sizeof(double));
	double *E_t = (double *)malloc(d * d * sizeof(double));

	// Center dataset
	dataset_partial_mean(s, s, d, img, mean);
	center_dataset(s, d, img, mean);

	// SVD
	SVD(s, d, img, U, D, E_t);

	// Set singular values after t^th one to zero
	cblas_dscal(d - t, 0.0, D + t, 1);

	// Compute Pt
	SVD_reconstruct_matrix(s, d, U, D, E_t, img);
	double *Pt = img;

	// Free SVD space
	free(U);
	free(D);
	free(E_t);

	// Compute covariance matrix
	multiply_matrices(Pt, d, s, 1, Pt, s, d, 0, St, 1);

	// Eigendecomposition of St
	double *L = (double *)malloc(d * sizeof(double));
	eigen_decomposition(d, St, L);
	double *E = St;
	reverse_matrix_columns(E, d, t, d, Et);
	free(St);
	free(L);

	// Obtain compressed data by projecting on first t principal components
	double *Pt_Et = (double *)malloc(s * t * sizeof(double));
	multiply_matrices(Pt, s, d, 0, Et, d, t, 0, Pt_Et, 1);
	multiply_matrices(Pt_Et, s, t, 0, Et, t, d, 1, img, 1);
	free(Pt_Et);
	decenter_dataset(s, d, img, mean);

	// Set style
	if (style == 0) {
		set_local_extremes(img, s, d, 0.0, 255.99);
	}
	else if (style == 1) {
		double local_min = DBL_MAX, local_max = DBL_MIN;
		get_local_extremes(img, s, d, &local_min, &local_max);
		if (global_min > local_min) global_min = local_min;
		if (global_max < local_max) global_max = global_max;
		rescale_image(img, s, d, global_min, global_max);
	}

	// Output img to JPEG
	write_matrix_to_JPEG("compressed_image.jpeg", img, s, d);

	// Free memory 
	free(mean);
	free(Et);
	free(img);
	return 0;
}