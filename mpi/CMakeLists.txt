cmake_minimum_required(VERSION 3.12)
project(ParallelPCA)

# Find Packages
find_package(MPI REQUIRED)
find_package(JPEG REQUIRED)
find_package(LAPACK REQUIRED)  # Use LAPACK instead of GSL

# Add the executable
add_executable(main main.c utils/jpeg_to_matrix.c)

# Link the executable with libraries
target_link_libraries(main PRIVATE ${MPI_C_LIBRARIES} ${JPEG_LIBRARIES} ${LAPACK_LIBRARIES})

target_include_directories(main PRIVATE ${MPI_INCLUDE_PATH} ${JPEG_INCLUDE_DIR} ${LAPACK_INCLUDE_DIRS})
