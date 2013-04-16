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
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "omp.h"

typedef struct {
	size_t nlines; /*!< The number of number for the scan*/
	char *infile; /*!< The filestem of the input file */
	char *outfile; /*!< The filename where the output will be stored */
	double timer_global;
	double timer_1;
	double timer_2;
	double timer_3;
	double timer_4;
	int nthreads, nlevels, nalloc;
	int *a; /* Store the calculated exclusive */
	int *b;
	int *c;
	int *d;
	char *f, *fb;
} params_t;

double gk_WClockSeconds(void) {
#ifdef __GNUC__
	struct timeval ctime;
	gettimeofday(&ctime, NULL );
	return (double) ctime.tv_sec + (double) .000001 * ctime.tv_usec;
#else
	return (double)time(NULL);
#endif
}
#define Tclear(tmr) (tmr = 0.0)
#define Tstart(tmr) (tmr -= gk_WClockSeconds())
#define Tstop(tmr)  (tmr += gk_WClockSeconds())
#define Tget(tmr)   (tmr)

#define EXTRA 999
#define MIN(a,b) (((a)<(b))?(a):(b))

void cmd_parse(int argc, char *argv[], params_t *par);
void Seq_Scan(params_t * par);
void WriteOut(params_t * par);
void cleanup(params_t *par);
void OMP_Scan(params_t *par);

void PrintAll(params_t *par);

int main(int argc, char *argv[]) {
	params_t par;
	double timer_scan, timer_serial;
	Tclear(par.timer_global);
	Tclear(par.timer_1);
	Tclear(par.timer_2);
	Tclear(par.timer_3);
	Tclear(par.timer_4);
	Tstart(par.timer_global);
	int k;
	printf("Scan - OMP_Scan\n");
	Tstart(par.timer_4);
	cmd_parse(argc, argv, &par);
	Tstop(par.timer_4);

	Seq_Scan(&par);
	timer_serial = par.timer_3;
	for (k = 0; k < EXTRA; ++k) {
		Seq_Scan(&par);
		timer_serial = MIN(timer_serial,par.timer_3);
	}
	par.timer_3 = timer_serial;



	memcpy(par.a, par.b, (par.nalloc + 1) * sizeof(int));

	OMP_Scan(&par);
	timer_scan = par.timer_2;
	for (k = 0; k < EXTRA; ++k) {
		memcpy(par.a, par.b, (par.nalloc + 1) * sizeof(int));
		OMP_Scan(&par);
		timer_scan = MIN(timer_scan,par.timer_2);
	}
	par.timer_2 = timer_scan;

//	PrintAll(&par);

	WriteOut(&par);

	Tstop(par.timer_global);

//	printf("  wclock         (sec): \t%.8lf\n", Tget(par.timer_global));
//	printf("  timer4  Init   (sec): \t%.8lf\n", Tget(par.timer_4));
	printf("  timer3  Serial (sec) on %d runs: \t%.8lf\n", 1 + EXTRA,
			Tget(par.timer_3));
	printf("  timer2  Scan   (sec) on %d runs: \t%.8lf\n", 1 + EXTRA,
			Tget(par.timer_2));

	cleanup(&par);
	return 0;
}

void OMP_Scan(params_t *par) {
	omp_set_num_threads(par->nthreads);
//	omp_set_num_threads(8);
	int nlevels = par->nlevels;
//	printf("nlevels=%d, nalloc=%d\n", par->nlevels, par->nalloc);
	int d, k, t;
	int step2d = 1, step2d1;

	/* ****************** UP SWEEP ******************************/
	Tclear(par->timer_2);
	Tstart(par->timer_2);
	for (d = 1; d < par->nlevels; ++d) {
		step2d1 = step2d * 2;
#pragma omp parallel for shared(par) schedule(static)
		for (k = 0; k < par->nalloc; k = k + step2d1) {
			par->a[k + step2d1 - 1] = par->a[k + step2d - 1]
					+ par->a[k + step2d1 - 1];
		}
		step2d = step2d * 2;
	}

	//			printf("TID: %d \t d=%d \t k=%d \t left=%d \t right=%d\n",
	//					omp_get_thread_num(), d, k, k + step2d - 1,
	//					k + step2d1 - 1);
	/* ****************** DOWN SWEEP ******************************/
	par->a[par->nalloc - 1] = 0;
	step2d = par->nalloc / 2;
	for (d = par->nlevels - 1; d >= 0; --d) {
		step2d1 = step2d * 2;
#pragma omp parallel for shared(par) schedule(static)
		for (k = 0; k < par->nalloc; k = k + step2d1) {
			t = par->a[k + step2d - 1];
			par->a[k + step2d - 1] = par->a[k + step2d1 - 1];
			par->a[k + step2d1 - 1] = t + par->a[k + step2d1 - 1];
		}

		step2d = step2d / 2;
	}
//#pragma omp parallel shared(par,step2d1,step2d)
//		{
//			int offset = omp_get_thread_num();
//			for (k = offset * step2d1; k < par->nalloc;
//					k += par->nthreads * step2d1) {
//				t = par->a[k + step2d - 1];
//				par->a[k + step2d - 1] = par->a[k + step2d1 - 1];
//				par->a[k + step2d1 - 1] = t + par->a[k + step2d1 - 1];
//			}
//			step2d = step2d / 2;
//		}
//	}
	par->a[par->nlines] = par->a[par->nlines - 1] + par->b[par->nlines - 1];
	par->c = &(par->a[1]);
	Tstop(par->timer_2);
}

void cmd_parse(int argc, char *argv[], params_t *par) {

	par->nthreads = strtol(argv[1], NULL, 10);
	par->infile = argv[2];
	size_t nlines, i;
	int x;
	FILE * fin;
	if (argc == 3)
		printf("%d argc: nth=%s,  infile=%s,  \t", argc, argv[1], argv[2]);
	else if (argc == 4)
		printf("%d argc: nth=%s,  infile=%s,  outfile=%s\t", argc, argv[1],
				argv[2], argv[3]);
	else if (argc <= 2)
		fprintf(stderr, "Wrong number of arguments. argc=%d", argc);
	else
		printf(
				"%d argc: nth=%s,  infile=%s,  outfile=%s, more arguments ignored\t",
				argc, argv[1], argv[2], argv[3]);

	if ((fin = fopen(par->infile, "r")) == NULL ) {
		fprintf(stderr, "ERROR: Failed to open '%s' for reading\n",
				par->infile);
		exit(1);
	}
	if (fscanf(fin, "%zu\n", &nlines) != 1) {
		fprintf(stderr, "ERROR: Failed to read first line of '%s'\n",
				par->infile);
		fclose(fin);
		exit(2);
	}
	par->nlines = nlines;
	printf("nlines=%d\n", (int) par->nlines);
	par->nlevels = (int) ceil(log(par->nlines) / M_LN2);
	par->nalloc = (int) (pow(2, par->nlevels));
	par->a = (int*) calloc((1 + par->nalloc), sizeof(int));
	par->b = (int*) calloc((1 + par->nalloc), sizeof(int));
	par->d = NULL;
	if (!par->a) {
		fprintf(stderr, "Fail to allocate memory, a is a null %p\n", par->a);
	}
	if (!par->b) {
		fprintf(stderr, "Fail to allocate memory, b is a null %p\n", par->a);
	}
	for (i = 0; i < nlines; ++i) {
		if (fscanf(fin, "%d\n", &x) != 1) {
			fprintf(stderr,
					"ERROR: Failed to read integer from line %zu/%zu in '%s'\n",
					i, nlines, par->infile);
			fclose(fin);
			exit(3);
		}
		par->b[i] = x;
	}
	fclose(fin);

	par->outfile = (char*) malloc(sizeof(char) * 256);
	if (argc == 3) {
		strcpy(par->outfile, "output.txt\0");
	} else {
		strcpy(par->outfile, argv[3]);
	}
	memcpy(par->a, par->b, (1 + par->nalloc) * (sizeof(int)));

	/* For Sscan Init */
	par->f = NULL;
	par->fb = NULL;
//	OMP_Sscan_Init(par);
}

void Seq_Scan(params_t * par) {
	int i;
	Tclear(par->timer_3);
	Tstart(par->timer_3);
	if (par->d!=NULL)
		free((void*)par->d);
	par->d = (int*) malloc(sizeof(int) * par->nlines);
	par->d[0] = par->b[0];
	for (i = 1; i < par->nlines; ++i) {
		par->d[i] = par->d[i - 1] + par->b[i];
	}
	Tstop(par->timer_3);
}

void PrintAll(params_t *par) {
	int k;
	for (k = 0; k < par->nlines; ++k) {
		if (par->c[k] == par->d[k]) {
//			printf("k=%2d,  a=%d,  b=%d, c=%d,  d=%d\n", k, par->a[k],
//					par->b[k], par->c[k], par->d[k]);
		} else {
			printf("k=%2d\t,  a=%d\t b=%d\t c=%d\t  d=%d\t  WRONG!!!\n", k,
					par->a[k], par->b[k], par->c[k], par->d[k]);
		}
	}
}

void WriteOut(params_t * par) {
	FILE * fout;
	if ((fout = fopen(par->outfile, "w")) == NULL ) {
		fprintf(stderr, "ERROR: Failed to open '%s' for writing\n",
				par->outfile);
		exit(1);
	}
	int i;
	for (i = 0; i < par->nlines; ++i) {
		fprintf(fout, "%d\n", par->c[i]);
	}
	fclose(fout);
}

void cleanup(params_t *par) {
	free((void*) par->a);
	free((void*) par->b);
	if (!par->f) {
		free((void*) par->f);
		free((void*) par->fb);
	}
}
