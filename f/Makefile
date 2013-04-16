#CC=gcc

OPTFLAGS =  -g -O3 -fno-strict-aliasing -D_GNU_SOURCE -DDEBUG
COPTIONS = -DLINUX -D_FILE_OFFSET_BITS=64 -std=c99 -Wall
           #-Wno-unused-function -Wno-unused-label \
           #-Wno-parentheses -Wsequence-point -Wno-unused-variable
           
#Standard Libraries
STDLIBS = -lm
CFLAGS = $(COPTIONS)  $(OPTFLAGS)
LIBS = $(STDLIBS)

CC=mpicc
#CC=/soft/openmpi/1.3/linux-x86_64/bin/mpicc


all: matvec_mpi.exe

matvec_mpi.exe: matvec_mpi.c Makefile
	$(CC) -o matvec_mpi matvec_mpi.c $(CFLAGS)   $(LIBS)
	

clean:
	rm -f matvec_mpi