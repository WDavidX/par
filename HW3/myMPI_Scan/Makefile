CC = mpicc
OPTFLAGS =  -g -O3 -fno-strict-aliasing -D_GNU_SOURCE
COPTIONS = -DLINUX -D_FILE_OFFSET_BITS=64 -std=c99 -Wall\
           -Wno-unused-function -Wno-unused-label -Wno-unused-variable\
           -Wno-parentheses -Wsequence-point
           
#OpenMP options
OMPOPTIONS = -fopenmp -D__OPENMP__

#Standard Libraries
STDLIBS = -lm

CFLAGS = $(COPTIONS)  $(OPTFLAGS)
LIBS = $(STDLIBS)


all: myMPI_Scan

myMPI_Scan: myMPI_Scan.c
	$(CC)  myMPI_Scan.c -o myMPI_Scan $(OPTFLAGS) $(COPTIONS) $(OMPOPTIONS)  $(LIBS)

clean:
	rm -f *.o myMPI_Scan