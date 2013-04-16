
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


all: mpiscan ompscan ompsscan

mpiscan: mpiscan.c
	mpicc  mpiscan.c -o mpiscan $(CFLAGS) $(OMPOPTIONS)  $(LIBS)
	echo " "


ompscan: ompscan.c 
	gcc  ompscan.c -o ompscan $(CFLAGS) $(OMPOPTIONS) $(LIBS)
	echo " "

ompsscan: ompsscan.c 
	gcc  ompsscan.c -o ompsscan $(CFLAGS) $(OMPOPTIONS) $(LIBS)
	echo " "

clean:
	rm -f *.o mpiscan ompscan ompsscan output.txt
