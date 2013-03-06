#!/usr/local/bin/tcsh

make



echo "pthread running"
./hello_pthreads.ex

echo " "
echo "cuda running"
./hello_cuda.ex 

echo " "
echo "openmp"
./hello_omp.ex 


echo " "
echo "mpi running"

mpirun --mca orte_rsh_agent "rsh" --hostfile ~/hosts -np 4 ./hello_mpi.ex