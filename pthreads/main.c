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
	double *et;
	double *st;
};

/* Global vars */
int thread_count;
int t, s;
pthread_mutex_t m;

int barrier_counter = 0;
pthread_mutex_t barrier_mutex;
pthread_cond_t barrier_cond_var;

void barrier()
{
	pthread_mutex_lock(&barrier_mutex);
	barrier_counter++;
	if (barrier_counter == thread_count)
	{
		barrier_counter = 0;
		pthread_cond_broadcast(&barrier_cond_var);
	}
	else
	{
		while (pthread_cond_wait(&barrier_cond_var, &barrier_mutex) != 0)
			;
	}
	pthread_mutex_unlock(&barrier_mutex);
}

void *PCA_round_one(void *arg)
{
	struct ThreadData *thread_data = (struct ThreadData *)arg;

	// Data goes from *local_img to *local_img + (sizeof(double) * local_s)
	int local_s = thread_data->thread_s;
	int d = thread_data->thread_d;
	double *local_img = thread_data->thread_img;
	long rank = thread_data->rank;
	double *mean = thread_data->mean;
	double *St = thread_data->st;
	double *Et = thread_data->et;

	// Center the dataset
	pthread_mutex_lock(&m);
	dataset_partial_mean(s, local_s, d, local_img, mean);
	pthread_mutex_unlock(&m);

	barrier(); // wait for all the threads to accumulate on mean

	center_dataset(local_s, d, local_img, mean);

	// SVD
	double *U_local = (double *)malloc(local_s * local_s * sizeof(double));
	double *D_local = (double *)malloc(d * sizeof(double));
	double *E_localT = (double *)malloc(d * d * sizeof(double));
	SVD(local_s, d, local_img, U_local, D_local, E_localT);

	// Set singular values after t^th one to zero
	cblas_dscal(d - t, 0.0, D_local + t, 1);

	// Compute Pt_local
	SVD_reconstruct_matrix(local_s, d, U_local, D_local, E_localT, local_img);
	double *Pt_local = local_img;

	// Free SVD space
	free(U_local);
	free(D_local);
	free(E_localT);

	// Compute St
	pthread_mutex_lock(&m);
	multiply_matrices(Pt_local, d, local_s, 1, Pt_local, local_s, d, 0, St, 0);
	pthread_mutex_unlock(&m);

	barrier(); // wait for all the threads to accumulate on St

	// Do eigendecomposition of St in thread 0
	if (rank == 0)
	{
		double *L = (double *)malloc(d * sizeof(double));
		eigen_decomposition(d, St, L);
		double *E = St;
		reverse_matrix_columns(E, d, t, d, Et);
		free(St);
		free(L);
	}
	barrier(); // wait for thread 0

	// Obtain Pp_local (written inside local_img) by projecting Pt_local on Et (first t columns of E)
	double *Pt_localEt = (double *)malloc(local_s * t * sizeof(double));
	multiply_matrices(Pt_local, local_s, d, 0, Et, d, t, 0, Pt_localEt, 1);
	multiply_matrices(Pt_localEt, local_s, t, 0, Et, t, d, 1, local_img, 1);
	free(Pt_localEt);
	decenter_dataset(local_s, d, local_img, mean);
	return NULL;
}

int main(int argc, char *argv[])
{
	long thread;
	pthread_t *thread_handles;
	pthread_mutex_init(&m, NULL);
	pthread_mutex_init(&barrier_mutex, NULL);
	pthread_cond_init(&barrier_cond_var, NULL);

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
	thread_handles = (pthread_t *)malloc(thread_count * sizeof(pthread_t));

	// Allocate space for thread data
	double *mean = (double *)calloc(d, sizeof(double));
	double *St = (double *)calloc(d * d, sizeof(double));
	double *Et = (double *)malloc(d * t * sizeof(double));

	// Set threads data
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
		data[thread].et = Et;
		pthread_create(&thread_handles[thread], NULL, PCA_round_one, (void *)&data[thread]);
	}

	// Wait for the threads to finish and join
	for (thread = 0; thread < thread_count; thread++)
	{
		pthread_join(thread_handles[thread], NULL);
	}

	// Output img to JPEG
	write_matrix_to_JPEG("endimg.jpeg", img, s, d);

	// Free memory and destroy mutexes and conditions
	free(mean);
	free(Et);
	free(img);
	free(thread_handles);
	pthread_mutex_destroy(&m);
	pthread_mutex_destroy(&barrier_mutex);
	pthread_cond_destroy(&barrier_cond_var);
	return 0;
}
