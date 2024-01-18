cmake_minimum_required(VERSION 3.12)
project(ParallelPCA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Find Packages
find_package(MPI REQUIRED)
find_package(JPEG REQUIRED)
find_package(LAPACKE REQUIRED)

# Add the executable
add_executable(main main.c utils/io_utils.c utils/la_utils.c)

# Link the executable with libraries
target_link_libraries(main PRIVATE ${MPI_C_LIBRARIES} ${JPEG_LIBRARIES} ${LAPACKE_LIBRARIES_DEP})

target_include_directories(main PRIVATE ${MPI_INCLUDE_PATH} ${JPEG_INCLUDE_DIR} ${LAPACKE_INCLUDE_DIRS_DEP})
