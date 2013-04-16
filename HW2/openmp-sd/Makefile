CC = gcc 
OPTFLAGS = -fopenmp -O3 -fno-strict-aliasing -D_GNU_SOURCE
COPTIONS = -DLINUX -D_FILE_OFFSET_BITS=64 -std=c99 -Wall\
           -Wno-unused-function -Wno-unused-label -Wno-unused-variable\
           -Wno-parentheses -Wsequence-point

#Standard Libraries
STDLIBS = -lm

CFLAGS = $(COPTIONS)  $(OPTFLAGS)
LIBS = $(STDLIBS)

SOURCE = sd.c getopt.c cmdline.c util.c
EXE = sd

all: $(EXE)

$(EXE): $(SOURCE)
	$(CC) $(CFLAGS) $(SOURCE)  -o $(EXE) $(LIBS) 

.PHONY: clean cleanall

clean:
	rm -rf *.o $(EXE)
	ls

cleanall:
	rm -rf *.o $(EXE) *.nbrs
	ls
