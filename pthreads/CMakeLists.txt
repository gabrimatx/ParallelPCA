cmake_minimum_required(VERSION 3.12)
project(ParallelPCA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Find Packages
find_package(JPEG REQUIRED)
find_package(LAPACKE REQUIRED)
find_package(CBLAS REQUIRED)

# Add the executable
add_executable(main main.c utils/io_utils.c utils/la_utils.c)

# Link the executable with libraries
target_link_libraries(main PRIVATE ${JPEG_LIBRARIES} ${LAPACKE_LIBRARIES_DEP} ${CBLAS_LIBRARIES})

target_include_directories(main PRIVATE ${JPEG_INCLUDE_DIR} ${LAPACKE_INCLUDE_DIRS_DEP})
