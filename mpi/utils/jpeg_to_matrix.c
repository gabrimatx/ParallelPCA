#include <stdio.h>
#include <jpeglib.h>
#include <gsl/gsl_matrix.h>

// Function to read from a JPEG file to a GSL matrix
gsl_matrix* read_JPEG_To_GSL(char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
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

    // Allocate memory for GSL matrix
    gsl_matrix* matrix = gsl_matrix_alloc(cinfo.output_height, cinfo.output_width);
    
    // Read JPEG data into GSL matrix
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW row_pointer[1];
        row_pointer[0] = (JSAMPROW)malloc(cinfo.output_width * cinfo.output_components);
        jpeg_read_scanlines(&cinfo, row_pointer, 1);

        // Copy data from row_pointer to GSL matrix
        for (int i = 0; i < cinfo.output_width; ++i) {
            gsl_matrix_set(matrix, cinfo.output_scanline - 1, i, row_pointer[0][i]);
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

// Function to write from a GSL matrix to a JPEG file
void write_GSL_to_JPEG(char* filename, gsl_matrix* matrix) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
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
    cinfo.image_width = matrix->size2;
    cinfo.image_height = matrix->size1;
    cinfo.input_components = 1;  // Grayscale image
    cinfo.in_color_space = JCS_GRAYSCALE;

    // Set default compression parameters
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 75, TRUE);

    // Start compression
    jpeg_start_compress(&cinfo, TRUE);

    // Write GSL matrix data to JPEG
    JSAMPROW row_pointer[1];

    for (int i = 0; i < cinfo.image_height; ++i) {
        row_pointer[0] = (JSAMPROW)malloc(cinfo.image_width);
        for (int j = 0; j < cinfo.image_width; ++j) {
            row_pointer[0][j] = (JSAMPLE)gsl_matrix_get(matrix, i, j);
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
