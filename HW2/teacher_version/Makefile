CC = gcc 
OPTFLAGS = -O3 -fno-strict-aliasing -D_GNU_SOURCE
COPTIONS = -DLINUX -D_FILE_OFFSET_BITS=64 -std=c99 -Wall\
           -Wno-unused-function -Wno-unused-label -Wno-unused-variable\
           -Wno-parentheses -Wsequence-point

#OpenMP options
OMPOPTIONS = -fopenmp -D__OPENMP__

#Standard Libraries
STDLIBS = -lgcc_s -lm

CFLAGS = $(COPTIONS) $(OMPOPTIONS) $(OPTFLAGS)
LIBS = $(STDLIBS)

SOURCE = getopt.c all_common.c all_cmdline.c util.c omp_nbrs.c omp_main.c
EXE = ompsd

default:
	$(CC) $(CFLAGS) $(SOURCE) -o $(EXE) $(LIBS) 
