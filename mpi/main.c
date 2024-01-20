#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <lapacke.h>
#include <cblas.h>
#include "utils/io_utils.h"
#include "utils/la_utils.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int my_rank;
    int comm_sz;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    double *img;
    int s;
    int local_s;
    int d;
    int t;
    int style = 0;
    const double DBL_MIN = -1e5;
    const double DBL_MAX = 1e5;
    double global_min = 0.0, global_max = 255.99;

    // Read input image
    if (my_rank == 0)
    {
        char *input_filename;

        // Ensure that the input filename is provided as a command-line argument
        if (argc != 3 && argc != 4)
        {
            printf("Usage: %s <input_filename.jpg> <n_components> <style (optional)>\n", argv[0]);
            return 1;
        }
        input_filename = argv[1];
        t = atoi(argv[2]);

        // Read from JPEG to matrix
        img = read_JPEG_to_matrix(input_filename, &s, &d);
        
        if (t > d) {printf("ERROR: the number of Principal Components (%d) cannot be greater than the numebr of columns of the image (%d).\n\n", t, d); return 1;}
        if (argc == 4) style = atoi(argv[3]);

        local_s = s / comm_sz;
    }
    // Start timing
    double start, finish;
    start = MPI_Wtime();

    // Broadcast matrix dimensions and number of components
    MPI_Bcast(&s, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&local_s, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&t, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&style, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter local matrices to nodes
    double *local_img = (double *)malloc(sizeof(double) * local_s * d);

    MPI_Scatter(img, local_s * d, MPI_DOUBLE,
                local_img, local_s * d, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    if (my_rank == 0) free(img);

    // Center the dataset
    double *partial_mean = (double *)calloc(d, sizeof(double));
    double *mean = (double *)malloc(d * sizeof(double));
    dataset_partial_mean(s, local_s, d, local_img, partial_mean);
    MPI_Allreduce(partial_mean, mean, d, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    center_dataset(local_s, d, local_img, mean);
    free(partial_mean);

    // Perform SVD
    double *U_local = (double *)malloc(local_s * local_s * sizeof(double));
    double *D_local = (double *)calloc(d, sizeof(double));
    double *E_localT = (double *)malloc(d * d * sizeof(double));
    SVD(local_s, d, local_img, U_local, D_local, E_localT);

    // Set singular values after the t-th one to 0
    cblas_dscal(d - t, 0.0, D_local + t, 1);

    // Compute Pt_local
    SVD_reconstruct_matrix(local_s, d, U_local, D_local, E_localT, local_img);
    double *Pt_local = local_img;

    // Free SVD space
    free(U_local);
    free(D_local);
    free(E_localT);

    // Compute St_local
    double *St_local = (double *)malloc(d * d * sizeof(double));
    multiply_matrices(Pt_local, d, local_s, 1, Pt_local, local_s, d, 0, St_local);

    // Compute St with reduce
    double *St;
    if (my_rank == 0)
    {
        St = (double *)malloc(d * d * sizeof(double));
    }
    MPI_Reduce(St_local, St, d * d, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    free(St_local);

    // Do eigendecomposition of St in process 0
    double *Et = (double *)malloc(d * t * sizeof(double));
    if (my_rank == 0)
    {
        double *L = (double *)malloc(d * sizeof(double));
        eigen_decomposition(d, St, L);
        double *E = St;
        reverse_matrix_columns(E, d, t, d, Et);
        free(St);
        free(L);
    }

    // Broadcast Et
    MPI_Bcast(Et, d * t, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Obtain Pp_local by projecting Pt_local on Et (first t columns of E)
    double *Pt_localEt = (double *)malloc(local_s * t * sizeof(double));
    double *Pp_local = (double *)malloc(local_s * d * sizeof(double));
    multiply_matrices(Pt_local, local_s, d, 0, Et, d, t, 0, Pt_localEt);
    multiply_matrices(Pt_localEt, local_s, t, 0, Et, t, d, 1, Pp_local);
    free(Pt_local);
    free(Pt_localEt);
    free(Et);

    // Add the mean back to Pp_local
    decenter_dataset(local_s, d, Pp_local, mean);
    free(mean);

    // Normalization

    if (style == 0) {
        set_local_extremes(Pp_local, local_s, d, 0.0, 255.99);
    }
    else if (style == 1) {
        double local_min = DBL_MAX, local_max = DBL_MIN;
		get_local_extremes(Pp_local, local_s, d, &local_min, &local_max);
        
		MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Bcast (&global_min, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast (&global_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		rescale_image(Pp_local, local_s, d, global_min, global_max);
    }

    // Gather all the Pp_local in Pp
    double *Pp = NULL;
    if (my_rank == 0) {
        Pp = (double *)malloc(local_s * comm_sz * d * sizeof(double));
    }
    MPI_Gather(Pp_local, local_s * d, MPI_DOUBLE, Pp, local_s * d, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free(Pp_local);

    // Stop timing
    finish = MPI_Wtime();
    printf("Process %d > Elapsed time = %e seconds\n", my_rank, finish - start);

    // Output Pp to JPEG
    if (my_rank == 0)
    {
        write_matrix_to_JPEG("output.jpeg", Pp, local_s * comm_sz, d);
        free(Pp);
    }

    MPI_Finalize();

    return 0;
}
