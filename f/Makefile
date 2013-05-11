CC = nvcc 
OPTFLAGS = -O3 
COPTIONS = -DLINUX -D_FILE_OFFSET_BITS=64 
OMPOPTIONS = --compiler-options -fopenmp --compiler-options -D__OPENMP__
COMPILEROPTS=   --compiler-options -Wall  --compiler-options  -Wno-unused-variable
#Standard Libraries
STDLIBS = -lgcc_s -lm
NVCCFLAGS= -arch sm_20

CFLAGS = $(COPTIONS)  $(OPTFLAGS) $(STDLIBS) $(OMPOPTIONS) $(NVCCFLAGS)
LIBS = $(STDLIBS)

CSOURCE= getopt.c all_common.c all_cmdline.c util.c  cuda_main.c
COBJ= getopt.o all_common.o all_cmdline.o util.o  cuda_main.o
CUDASOURCE= cuda_nbrs.cu 
CUDAOBJ=cuda_nbrs.o
EXE = cssd

all: cssd

remake: cssd
	rm -f $(EXE) a.out *.o 

getopt.o: getopt.c Makefile
	$(CC) -c  getopt.c $(CFLAGS)
	
all_common.o: all_common.c Makefile
	$(CC) -c  all_common.c $(CFLAGS)

all_cmdline.o: all_cmdline.c
	$(CC) -c  all_cmdline.c	 $(CFLAGS)
	
util.o: util.c Makefile
	$(CC) -c  util.c $(CFLAGS)
	 
cuda_main.o: cuda_main.c Makefile
	$(CC) -c  cuda_main.c	 $(CFLAGS)
	
cuda_nbrs.o:cuda_nbrs.cu Makefile
	$(CC) -c  cuda_nbrs.cu $(CFLAGS)

	
$(EXE): getopt.o all_common.o all_cmdline.o util.o  cuda_main.o  cuda_nbrs.o
	$(CC) getopt.o all_common.o all_cmdline.o util.o  cuda_main.o  cuda_nbrs.o -o $(EXE) $(CFLAGS)

.PHONY: clean remake
	
clean:
	rm -f $(EXE) a.out *.o 

