CC = gcc 
OPTFLAGS =  -g -O3 -fno-strict-aliasing -D_GNU_SOURCE
COPTIONS = -DLINUX -D_FILE_OFFSET_BITS=64 -std=c99 -Wall\
           -Wno-unused-function -Wno-unused-label -Wno-unused-variable\
           -Wno-parentheses -Wsequence-point

#Standard Libraries
STDLIBS = -lm

CFLAGS = $(COPTIONS)  $(OPTFLAGS)
LIBS = $(STDLIBS)

SOURCE = SerScan.c
HEADER= SerScan.h
EXE = SerScan

all: $(SOURCE) $(HEADER)
	$(CC)  $(OPTFLAGS) $(COPTIONS) $(SOURCE) $(HEADER) -o $(EXE) $(LIBS)

clean:
	rm -f *.o $(EXE)