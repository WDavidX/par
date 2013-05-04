CC = nvcc 
OPTFLAGS = -O3 -D_GNU_SOURCE -g
COPTIONS =  -D_FILE_OFFSET_BITS=64 

#Standard Libraries
STDLIBS = -lgcc_s -lm
INCLUDEDIR = -I~/par/f

CFLAGS = $(COPTIONS)  $(OPTFLAGS)
LIBS = $(STDLIBS)

PRESRC=  all_common.c all_cmdline.c getopt.c util.c
NEWSRC =   main_cuda.c   nbrs_cuda.cu  
SOURCE =  $(PRESRC) $(NEWSRC)
EXE = cssd

all:  $(SOURCE)
	$(CC) $(CFLAGS)  $(SOURCE) -o $(EXE) $(LIBS) $(INCLUDEDIR)

.PHONY: clean

clean:
	rm -f $(EXE)