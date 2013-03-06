CC = gcc 
OPTFLAGS = -pthread -O3 -fno-strict-aliasing -D_GNU_SOURCE -g
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

$(EXE):
	$(CC) $(CFLAGS) $(SOURCE) $(LIBS) -o $(EXE)

.PHONY: clean cleanall

clean:
	rm -rf *.o $(EXE)
	ls

cleanall:
	rm -rf *.o $(EXE) *.nbrs
	ls
	
