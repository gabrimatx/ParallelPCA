#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include <cblas.h>
#include "utils/io_utils.h"
#include "utils/la_utils.h"
#include <time.h>

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
	if (t > d)
	{
		printf("ERROR: the number of Principal Components (%d) cannot be greater than the number of columns of the image (%d).\n\n", t, d);
		return 1;
	}

	int style = 0;
	if (argc == 4)
		style = atoi(argv[3]);

	// Start timer
	clock_t start_time = clock();

	// Allocate space needed
	double *mean = (double *)calloc(d, sizeof(double));
	double *U = (double *)malloc(s * s * sizeof(double));
	double *D = (double *)malloc(d * sizeof(double));
	double *ET = (double *)malloc(d * d * sizeof(double));

	// Center dataset
	dataset_partial_mean(s, s, d, img, mean);
	center_dataset(s, d, img, mean);

	// SVD
	SVD(s, d, img, U, D, ET);

	// Set singular values after t^th one to zero
	cblas_dscal(d - t, 0.0, D + t, 1);

	// Compute img compressed
	SVD_reconstruct_matrix(s, d, U, D, ET, img);

	// Free SVD space
	free(U);
	free(D);
	free(ET);
	decenter_dataset(s, d, img, mean);

	// Set style
	if (style == 0)
	{
		set_local_extremes(img, s, d, 0.0, 255.99);
	}
	else if (style == 1)
	{
		double local_min = DBL_MAX, local_max = DBL_MIN;
		get_local_extremes(img, s, d, &local_min, &local_max);
		if (global_min > local_min)
			global_min = local_min;
		if (global_max < local_max)
			global_max = local_max;
		rescale_image(img, s, d, global_min, global_max);
	}

	// Record the end time
	clock_t end_time = clock();

	// Output img to JPEG
	write_matrix_to_JPEG("compressed_image.jpg", img, s, d);

	// Free memory
	free(mean);
	free(img);


	// Calculate the execution time in seconds
	double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

	// Print the execution time
	printf("Execution Time: %f seconds\n", execution_time);

	return 0;
}