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

#define EXTRA 99
#define MIN(a,b) (((a)<(b))?(a):(b))

void cmd_parse(int argc, char *argv[], params_t *par);
void WriteOut(params_t * par);
void cleanup(params_t *par);

void OMP_Sscan(params_t *par);
void OMP_Sscan_Init(params_t *par);
void PrintAll(params_t *par);

int main(int argc, char *argv[]) {
	params_t par;
	double timer_sscan;
	Tclear(par.timer_global);
	Tclear(par.timer_1);
	Tclear(par.timer_2);
	Tclear(par.timer_3);
	Tclear(par.timer_4);
	Tstart(par.timer_global);
	int k;
	printf("\nScan - OMP_Sscan\n");
	Tstart(par.timer_4);
	cmd_parse(argc, argv, &par);
	Tstop(par.timer_4);

	OMP_Sscan(&par);
	timer_sscan = par.timer_1;
	for (k = 0; k < EXTRA; ++k) {
		memcpy(par.a, par.b, sizeof(int) * (1 + par.nalloc));
		memcpy(par.f, par.fb, (1 + par.nalloc) * (sizeof(char)));
		OMP_Sscan(&par);
		timer_sscan = MIN(par.timer_1,timer_sscan);
	}
	par.timer_1 = timer_sscan;
//	PrintAll(&par);
	WriteOut(&par);
	Tstop(par.timer_global);


	printf("  wclock         (sec): \t%.8lf\n", Tget(par.timer_global));
	printf("  timer4  Init   (sec): \t%.8lf\n", Tget(par.timer_4));
	printf("  timer1  Sscan  (sec) on %d runs: \t%.8lf\n", 1 + EXTRA,
			Tget(par.timer_1));
	cleanup(&par);
	return 0;
}

void OMP_Sscan_Init(params_t *par) {

	if (par->f != NULL ) {
		fprintf(stderr, "ERROR: Flag array pointer is not NULL\n");
		exit(5);
	}
	par->f = (char*) calloc((1 + par->nalloc), sizeof(char));
	if (par->f == NULL ) {
		fprintf(stderr, "ERROR: Fail to create flag array\n");
		exit(6);
	}
	if (par->fb != NULL ) {
		fprintf(stderr, "ERROR: Flag array pointer is not NULL\n");
		exit(5);
	}
	par->fb = (char*) malloc((1 + par->nalloc) * sizeof(char));
	if (par->fb == NULL ) {
		fprintf(stderr, "ERROR: Fail to create flag backup array\n");
		exit(6);
	}
	par->f[0] = 1;
	int k, i, ct = 0;
	srand(time(NULL ));

	if (par->nlines > 1){
		printf("Sscan_INIT, generating %d entries with flag 1\n",
					1 + (int) floor(sqrt(par->nlines)));
		while (ct < (int) floor(sqrt(par->nlines))) {
			int r = rand();
			i = rand() % par->nlines;
			if (par->f[i] == 0) {
				par->f[i] = 1;
				++ct;
			}

		}
	}else{
		printf("Sscan_INIT, generating %d entries with flag 1\n",
					1 );
	}
	memcpy(par->fb, par->f, (1 + par->nalloc) * (sizeof(char)));
}

void OMP_Sscan(params_t *par) {
	omp_set_num_threads(par->nthreads);
	int nlevels = (int) ceil(log(par->nlines) / M_LN2);
	int d, k, t;
	int levelstep2d = 1, levelstep2d1;

	/* ****************** UP SWEEP ******************************/
	Tclear(par->timer_1);
	Tstart(par->timer_1);
	for (d = 1; d < par->nlevels; ++d) {
		levelstep2d1 = levelstep2d * 2;
#pragma omp parallel for shared(par,levelstep2d,levelstep2d1) \
		schedule(static)
		for (k = 0; k < par->nalloc; k = k + levelstep2d1) {
			if (!par->f[k + levelstep2d1 - 1])
				par->a[k + levelstep2d1 - 1] = par->a[k + levelstep2d - 1]
						+ par->a[k + levelstep2d1 - 1];
			par->f[k + levelstep2d1 - 1] = par->f[k + levelstep2d - 1]
					| par->f[k + levelstep2d1 - 1];
		}
		levelstep2d = levelstep2d * 2;
	}

	/* ****************** DOWN SWEEP ******************************/
	par->a[par->nalloc - 1] = 0;

	levelstep2d = par->nalloc / 2;

	for (d = par->nlevels - 1; d >= 0; --d) {
		levelstep2d1 = levelstep2d * 2;
#pragma omp parallel for shared(par,levelstep2d,levelstep2d1) \
		schedule(static)
		for (k = 0; k < par->nalloc; k = k + levelstep2d1) {
			t = par->a[k + levelstep2d - 1];
			par->a[k + levelstep2d - 1] = par->a[k + levelstep2d1 - 1];

			if (par->fb[k + levelstep2d]) {
				par->a[k + levelstep2d1 - 1] = 0;
			} else if (par->f[k + levelstep2d - 1]) {
				par->a[k + levelstep2d1 - 1] = t;
			} else {
				par->a[k + levelstep2d1 - 1] = t + par->a[k + levelstep2d1 - 1];
			}
			par->f[k + levelstep2d - 1] = 0;
		}
		levelstep2d = levelstep2d / 2;
	}

#pragma omp parallel for shared(par,levelstep2d,levelstep2d1) \
		schedule(static)
	for (k = 0; k < par->nlines; k = k + 1) {
		par->a[k] = par->b[k] + par->a[k];
	}
	Tstop(par->timer_1);
	par->c = par->a;

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
	OMP_Sscan_Init(par);
}

void WriteOut(params_t * par) {
	FILE * fout;
	printf("\nWriteout nlines=%d\n", (int) par->nlines);
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

void PrintAll(params_t *par) {
	int k;
	for (k = 0; k < par->nlines; ++k) {
		printf("k=%d\t b=%d\t a=%3d\t  fold=%d, fmod=%d\n", k, par->b[k],
				par->a[k], par->fb[k], par->f[k]);
	}
}
