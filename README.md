# ParallelPCA
## Introduction
ParallelPCA is a program that implements a distributed algorithm for Principal Components Analysis. The two methodologies implemented are multithreading and message passing between multiple processes. The input and output is performed with grayscale images for visualization and simplicity, but can be easily applied to RGB images and other objects.
## Installation
### Clone 
```bash
# PARALLEL_PCA=/path/to/clone/ParallelPCA
git clone https://github.com/rubenciranni/ParallelPCA.git $PARALLEL_PCA
```
### Install dependencies
The following packages are required to run the application:
```bash
cmake
mpi
pthreads
jpeg
lapacke
cblas
```
## How to build
Inside any of the `pthreads/`, `mpi/`, `serial/` directories:
```bash
mkdir build
cd build
cmake ..
make
```
## How to run
### Pthreads
```bash
./main <n_threads> <image path> <n_principal_components> <style (optional)>
```
### MPI
```bash
mpirun -n <number of processes> ./main <image path> <number of components> <normalization style (optional)>
```
### Serial
```bash
./main <image path> <n_components> <style (optional)>
```
## Authors
We're a group of three hard-working university students at Sapienza University of Rome. ParallelPCA is a project that we have undertaken as part of our academic curriculum, specifically designed to fulfill the requirements of one of our examinations -  Embedded and Multicore Systems Programming.
For any clarifications or further information regarding the project, please feel free to reach out to us.

<img src=".\figs\AUTHORS.svg">

## License
ParallelPCA is released under the [MIT License](./LICENSE). 
