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

SOURCE = myMPI_Scan.c
HEADER= myMPI_Scan.h
EXE = myMPI_Scan

all: $(SOURCE) $(HEADER)
	$(CC)  $(OPTFLAGS) $(COPTIONS) $(OMPOPTIONS) $(SOURCE) $(HEADER) -o $(EXE) $(LIBS)

clean:
	rm -f *.o $(EXE)