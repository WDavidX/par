CC=gcc

all: hello_pthreads.ex hello_mpi.ex hello_omp.ex

hello_pthreads.ex:
	$(CC) -pthread -o hello_pthreads.ex hello_pthreads.c 
	
hello_mpi.ex:
	mpicc -o hello_mpi.ex hello_mpi.c 

hello_cuda.ex:
	nvcc -o hello_cuda.ex hello_cuda.cu 
	
hello_omp.ex:
	$(CC) -fopenmp -o hello_omp.ex hello_omp.c
	
clean:
	rm -f hello_pthreads.ex hello_mpi.ex hello_cuda.ex hello_omp.ex
