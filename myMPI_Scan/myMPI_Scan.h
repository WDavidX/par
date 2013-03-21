#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <errno.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <signal.h>
#include <setjmp.h>
#include <assert.h>
#include <inttypes.h>
//#include <sys/resource.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

typedef struct {
	double timer_global;
	double timer_1;
	double timer_2;
	double timer_3;
	double timer_4;
	int n, nalloc, nlevels;
	int *a;	// copied array for OMP_Scan
	int *b; // copied array for myOMP_Scan
	int *c; // back of of the original array
	int num_procs,myid, name_len;
	char *proc_name;
} params_t;

