#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

double *read_JPEG_to_matrix(char *filename, int *rows, int *cols)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        perror("Error opening file");
        return NULL;
    }

    // Initialize JPEG decompression object
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    // Set the source file
    jpeg_stdio_src(&cinfo, file);

    // Read JPEG header
    jpeg_read_header(&cinfo, TRUE);

    // Start decompression
    jpeg_start_decompress(&cinfo);

    // Set the dimensions
    *rows = cinfo.output_height;
    *cols = cinfo.output_width;

    // Allocate memory for the unrolled matrix
    double *matrix = (double *)malloc(sizeof(double) * (*rows) * (*cols));

    // Read JPEG data into the matrix
    while (cinfo.output_scanline < cinfo.output_height)
    {
        JSAMPROW row_pointer[1];
        row_pointer[0] = (JSAMPROW)malloc(cinfo.output_width * cinfo.output_components);
        jpeg_read_scanlines(&cinfo, row_pointer, 1);

        // Copy data from row_pointer to matrix
        for (int i = 0; i < cinfo.output_width; ++i)
        {
            matrix[(cinfo.output_scanline - 1) * (*cols) + i] = (double)row_pointer[0][i];
        }

        free(row_pointer[0]);
    }

    // Finish decompression
    jpeg_finish_decompress(&cinfo);

    // Clean up
    jpeg_destroy_decompress(&cinfo);
    fclose(file);

    return matrix;
}

void write_matrix_to_JPEG(char *filename, double *matrix, int rows, int cols)
{
    FILE *file = fopen(filename, "wb");
    if (!file)
    {
        perror("Error opening file");
        return;
    }

    // Initialize JPEG compression object
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    // Set the destination file
    jpeg_stdio_dest(&cinfo, file);

    // Set image parameters
    cinfo.image_width = cols;
    cinfo.image_height = rows;
    cinfo.input_components = 1; // Grayscale image
    cinfo.in_color_space = JCS_GRAYSCALE;

    // Set default compression parameters
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 75, TRUE);

    // Start compression
    jpeg_start_compress(&cinfo, TRUE);

    // Write matrix data to JPEG
    JSAMPROW row_pointer[1];

    for (int i = 0; i < rows; ++i)
    {
        row_pointer[0] = (JSAMPROW)malloc(cols);
        for (int j = 0; j < cols; ++j)
        {
            row_pointer[0][j] = (JSAMPLE)matrix[i * cols + j];
        }
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
        free(row_pointer[0]);
    }

    // Finish compression
    jpeg_finish_compress(&cinfo);

    // Clean up
    jpeg_destroy_compress(&cinfo);
    fclose(file);
}

void print_matrix(char *name, int rows, int cols, double *A)
{
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%.2f\t", A[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_matrix_int(char *name, int rows, int cols, double *A)
{
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d\t", (int)A[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_vector(char *name, int dim, double *v)
{
    printf("%s:\n", name);
    for (int i = 0; i < dim; i++)
        printf("%f\n", v[i]);
    printf("\n");
}

void autotester(char *filename, int rows, int cols, double *A)
{
    double *test = (double *)malloc(rows * cols * sizeof(double));

    printf("%s:\n", filename);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            test[i * rows + j] = A[i * cols + j];

    write_matrix_to_JPEG(filename, test, rows, cols);

    free(test);
}
