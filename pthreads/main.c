#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <lapacke.h>
#include <cblas.h>
#include "utils/io_utils.h"
#include "utils/la_utils.h"

/* Thread data */
struct ThreadData 
{
    double *thread_img;
    int thread_s;
    int thread_d;
    long rank;
    double *mean;
	double *st;
};


/* Global var */
int thread_count;
int t, s;
pthread_mutex_t m;

void *LocalPCA (void* arg){
 	struct ThreadData *thread_data = (struct ThreadData *)arg;

 	// Data goes from *local_img to *local_img + (sizeof(double) * local_s)
 	int local_s = thread_data->thread_s;
 	int d = thread_data->thread_d;
 	double *local_img = thread_data->thread_img; 
 	long rank = thread_data->rank;
 	double *mean = thread_data->mean;
	double *St = thread_data->st;

	// Center the dataset
	
	pthread_mutex_lock(&m);
	dataset_partial_mean(s, local_s, d, local_img, mean);
	center_dataset(local_s, d, local_img, mean);
	pthread_mutex_unlock(&m);
	

 	// SVDod
	pthread_mutex_lock(&m);
 	double *U_local = (double *)malloc(local_s * local_s * sizeof(double));
    double *D_local = (double *)malloc(d * sizeof(double));
    double *E_localT = (double *)malloc(d * d * sizeof(double));
    SVD(local_s, d, local_img, U_local, D_local, E_localT);
	pthread_mutex_unlock(&m);

    // Set singular values after t^th one to zero
    cblas_dscal(d - t, 0.0, D_local + t, 1);

    // Compute Pt_local
    SVD_reconstruct_matrix(local_s, d, U_local, D_local, E_localT, local_img);

	// Compute Covariance matrix
	/*
	for (int i = 0; i < d; i++) {
		for (int j = 0; j < d; j++){
			double c = 0;
			for (int k = 0; k < local_s; k++){
				c += local_img[i + k * d] * local_img[j + k * d];
			
			} 
			pthread_mutex_lock(&m);
				St[i*d + j] += c;
			pthread_mutex_unlock(&m);
		}
	}
	*/
	char output_filename[20];
    sprintf(output_filename, "output%ld.jpeg", rank);
    write_matrix_to_JPEG(output_filename, local_img, local_s, d);

	double *Pt_local = local_img;
	pthread_mutex_lock(&m);
	multiply_matrices(Pt_local, d, local_s, 1, Pt_local, local_s, d, 0, St);
	pthread_mutex_unlock(&m);




	printf("thread #%ld: reconstruct matrix passed\n", rank);
	

    // Output custom matrices to JPEG (to debug)


	return NULL;
}

int main(int argc, char* argv[]) {
	long thread;
	pthread_t* thread_handles;
	pthread_mutex_init(&m, NULL);

	// Get number of threads from the command line
	thread_count = strtol(argv[1], NULL, 10);

	// Ensure that the input filename is provided as a command-line argument
	char *input_filename;
	if (argc != 4)
	{
		printf("Usage: %s <input_filename.jpg> <n_threads>\n", argv[0]);
		return 1;
	}
	input_filename = argv[2];
	double *img;
	int d;
	img = read_JPEG_to_matrix(input_filename, &s, &d);
	t = atoi(argv[3]);

	// Allocate threads
	thread_handles = malloc (thread_count * sizeof(pthread_t));

	// Allocate space for thread returns
	double *St = calloc(d * d, sizeof(double));
	double *mean = calloc(d, sizeof(double));	

	// image batch size := s * d / # of threads
	// Split image between threads

	struct ThreadData data[thread_count];
	int offset = (s / thread_count) * d;
	for (thread = 0; thread < thread_count; thread++)
	{
		data[thread].thread_img = img + (offset * thread);
		data[thread].thread_s = s / thread_count;
		data[thread].thread_d = d;
		data[thread].rank = thread;
		data[thread].mean = mean;
		data[thread].st = St; 
		pthread_create(&thread_handles[thread], NULL, LocalPCA, (void*)&data[thread]);
	}

	// Wait for the threads
	for (thread = 0; thread < thread_count; thread++)
	{ 
		pthread_join(thread_handles[thread], NULL);
	}

	// Fine round 1

	// debug
	write_matrix_to_JPEG("endimg.jpeg", img, s, d);
	print_matrix("Covariance St", d, d, St);
	printf("hello from the main thread!\n");

	// Free memory and delete mutex
	free(thread_handles);
	pthread_mutex_destroy(&m);
	return 0;

}
