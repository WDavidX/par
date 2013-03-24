CC = gcc 
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

all:  OMP_Sscan

OMP_Sscan: OMP_Sscan.c 
	$(CC)  OMP_Sscan.c -o OMP_Sscan $(CFLAGS) $(OMPOPTIONS) $(LIBS)


clean:
	rm -f *.o OMP_Scan output.txt OMP_Sscan
