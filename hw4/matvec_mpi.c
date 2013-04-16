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
#include <mpi.h>

/* Help functions */
int TextLines(char *namestr);
#define LTERM                   (void **) 0     /* List terminator for GKfree() */
void gk_free(void **ptr1, ...);

/* Macros */
#define Tclear(tmr) (tmr = 0.0)
#define Tstart(tmr) (tmr -= MPI_Wtime())
#define Tstop(tmr)  (tmr += MPI_Wtime())
#define Tget(tmr)   (tmr)

#ifdef DEBUG
#define DPRINTF(fmt, args...)    fprintf(stdout, fmt, ## args)
#else
#define DPRINTF(fmt, args...)    /* Don't do anything in release builds */
#endif

#define wcalloc(PTR,TYPE, ALLOCNUM,ERRMSG)\
	do{\
	PTR=(TYPE*)malloc(sizeof(TYPE)*(ALLOCNUM));\
	if ((PTR)==NULL)\
		fprintf(stderr,"Malloc failure PTR=%s @ Line %d\t %s\n",#PTR,__LINE__,ERRMSG);\
	}while(0)

#define wccalloc(PTR,TYPE, ALLOCNUM,ERRMSG)\
	do{\
	PTR=(TYPE*)calloc((ALLOCNUM),sizeof(TYPE));\
	if ((PTR)==NULL)\
		fprintf(stderr,"Malloc failure PTR=%s @ Line %d\t %s\n",#PTR,__LINE__,ERRMSG);\
	}while(0)

#define wcint(a,n,offset)\
	do{ int local_k;\
		for (local_k=0;local_k<n;++local_k)\
			printf("%s[%d] = %d\n",#a,local_k+offset,(a[local_k]));\
	}while(0)
#define wcdouble(a,n,offset)\
	do{ int local_k;\
		for (local_k=0;local_k<n;++local_k)\
			printf("%s[%d] = %5.3lf\n",#a,local_k+offset,(a[local_k]));\
	}while(0)

#define wcfree(PTR,ERRMSG)\
	do{\
	if (PTR==NULL)\
		fprintf(stderr,"Empty pointer freed PTR=%s @ Line %d\t %s\n",#PTR,__LINE__,ERRMSG);\
else{\
	free((void*) PTR);}\
	}while(0)

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#ifdef DEBUG
#define PASSHERE(x) do{fprintf(stderr,"Pass line %d    %s\n",__LINE__,x);}while(0)
#else
#define PASSHERE(x) do{}while(0)
#endif

typedef struct {
	double t0, t1, t2, t3, t10;
	int np, id;
	char proc_name[MPI_MAX_PROCESSOR_NAME];
	int namelength;
	char *fmat, *fvec, *fout;
	int *blkstart;
	int blkconstrow;
	int nnz, nrow;
} par_t;

typedef struct {
	int nrow, nnz, blknrow, blknnz;
	double *tmpb, *b;				//tmpb: tmp save b; before inspection;
	int bct, tmpbct,browstart;		//number of element in tmpb
} mat_t;

/* Global varibles */
par_t par;
mat_t mat;
int broadcastarray[2] = { -1, -1 };

int Step1(par_t *par, mat_t *mat);
int Step1other(par_t *par, mat_t *mat);

int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv); /* starts MPI */
	par.t0 = par.t1 = par.t2 = par.t3 = par.t10 = 0;
	Tstart(par.t0);
	MPI_Comm_rank(MPI_COMM_WORLD, &par.id); /* get current process id */
	MPI_Comm_size(MPI_COMM_WORLD, &par.np); /* get number of processes */
	MPI_Get_processor_name(par.proc_name, &par.namelength);
	if (par.id == 0) {
		par.fmat = argv[1];
		par.fvec = argv[2];
		par.fout = NULL;
	} else {
		par.fmat = NULL;
		par.fmat = NULL;
		par.fout = NULL;
	}
	printf("Proc %d of %d, %s, fmat=%s, fvec=%s\n", par.id, par.np,
			par.proc_name, par.fmat, par.fvec);
	MPI_Barrier(MPI_COMM_WORLD);
	if (par.id == 0) {
		Step1(&par, &mat);
	} else {
		Step1other(&par, &mat);
	}

	if (par.id == 0) {
		Tstop(par.t0);
		printf("Overall time         t0   (sec):\t\t%.8lf\n", par.t0);
		printf("Total time taken     t1   (sec):\t\t%.8lf\n", par.t1);
		printf("Time taken (step5&6) t2   (sec):\t\t%.8lf\n", par.t2);
		printf("Time taken (fileio) t10   (sec):\t\t%.8lf\n", par.t10);
	}

	MPI_Finalize();

	return 0;
}

int Step1(par_t *par, mat_t *mat) {
	Tstart(par->t10);
	par->nnz = TextLines(par->fmat);
	par->nrow = TextLines(par->fvec);
//	par->nnz =29599380;
//	par->nrow=400000;
	broadcastarray[0] = par->nnz;
	broadcastarray[1] = par->nrow;
	MPI_Bcast(&broadcastarray, 2, MPI_INT, par->id, MPI_COMM_WORLD);

	Tstop(par->t10);
	par->blkconstrow = par->nrow / par->np;
	wccalloc(par->blkstart, int, par->np + 1, "blkstart");
	par->fout = (char*) malloc(sizeof(char) * 256);
	sprintf(par->fout, "./o%d.vec", par->nrow);
	DPRINTF("%s generates output file %s\n",__FILE__,par->fout);
	int *mati, *matj;
	double *matval;
	double *vecb;
	wcalloc(mati, int, par->nnz, "mati");
	wcalloc(matj, int, par->nnz, "matj");
	wcalloc(matval, double, par->nnz, "matval");
	wcalloc(vecb, double, par->nrow, "vecval");
	char *getblkfirst = (char*) calloc(par->nrow + 1, sizeof(char));

	int k;
	FILE *fA, *fb;
	Tstart(par->t10);
	if ((fA = fopen(par->fmat, "r")) == NULL
			|| (fb = fopen(par->fvec, "r")) == NULL ) {
		fprintf(stderr, "Fail to open file\n");
		exit(1);
	}

	int blkcounter = 0;
	for (k = 0; k < par->nrow; ++k) {
		fscanf(fb, "%lf", &vecb[k]);
//		printf("%.4lf\n",vecb[k]);
	}

	PASSHERE("PRINT B");
	wcdouble(vecb,par->nrow,0);

	for (blkcounter = 1; blkcounter < par->np - 1; ++blkcounter) {
		MPI_Send(&vecb[blkcounter * par->blkconstrow], par->blkconstrow,
				MPI_DOUBLE, blkcounter, 0, MPI_COMM_WORLD);
	}
	MPI_Send(&vecb[blkcounter * par->blkconstrow],
			par->nrow - (par->np - 1) * par->blkconstrow, MPI_DOUBLE, blkcounter,
			0, MPI_COMM_WORLD);
	mat->tmpbct=par->blkconstrow;
	wcalloc(mat->tmpb,double,mat->tmpbct,"");
	memcpy(mat->tmpb,vecb,sizeof(double)*mat->tmpbct);
	wcdouble(mat->tmpb,mat->tmpbct,mat->browstart);
	blkcounter = 0;


	for (k = 0; k < par->nnz; ++k) {
		fscanf(fA, "%d %d %lf", &mati[k], &matj[k], &matval[k]);
		if (blkcounter < par->np && getblkfirst[mati[k]] == 0
				&& (mati[k] % par->blkconstrow) == 0) {
			/* detect start of a new block */
			par->blkstart[blkcounter] = k;
			blkcounter++;
			getblkfirst[mati[k]] = 1;
		}
	}DPRINTF("blksize %d\t blkcounter %d\n",par->blkconstrow,blkcounter);
	par->blkstart[blkcounter] = k;

	for (k = 0; k < par->np; ++k) {
		DPRINTF("BLK # %d, blkstart %8d, blksize %8d\n", k, par->blkstart[k],
				par->blkstart[k + 1] - par->blkstart[k]);
	}

	fclose(fA);
	fclose(fb);
	Tstop(par->t10);
	return 0;

}

int Step1other(par_t *par, mat_t *mat) {
	MPI_Status status;
	MPI_Bcast(&broadcastarray, 2, MPI_INT, 0, MPI_COMM_WORLD);
	DPRINTF("NNZ=%d, NROW=%d, id=%d\n",broadcastarray[0],broadcastarray[1],par->id);
	par->nnz = broadcastarray[0];
	par->nrow = broadcastarray[1];
	par->blkconstrow = par->nrow / par->np;
	if (par->id != par->np - 1) {
		mat->tmpbct = par->blkconstrow;
	} else {
		mat->tmpbct = par->nrow - par->blkconstrow * (par->np - 1);
		printf("mat->tmpbct=%d\n",mat->tmpbct);
	}
	mat->browstart=par->id*par->blkconstrow;
	wcalloc(mat->tmpb, double, mat->tmpbct, "create temp b");
	printf("********************%d**\n",par->id);
	MPI_Recv(mat->tmpb, mat->tmpbct, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD,
			&status);
	wcdouble(mat->tmpb,mat->tmpbct,mat->browstart);
//	for (int kk=0;kk<mat->tmpbct;++kk){
//		printf("%d %lf\n",kk,mat->tmpb[kk]);
//	}
	return 0;
}

int TextLines(char *namestr) {
	FILE *f;
	if ((f = fopen(namestr, "r")) == NULL ) {
		printf("Fail to open file %s\n", namestr);
		exit(1);
	}
	char ch, prev = '\n';
	int nlines = 0;
	do {
		ch = fgetc(f);
		if (ch == '\n' && prev != '\n')
			nlines++;
		prev = ch;
	} while (ch != EOF);
	fclose(f);
	printf("Number of lines in %s \t = %d\n", namestr, nlines);
	return nlines;
}

/*************************************************************************
 * This function is my wrapper around free, allows multiple pointers
 **************************************************************************/
void gk_free(void **ptr1, ...) {
	va_list plist;
	void **ptr;

	if (*ptr1 != NULL )
#ifdef USE_DLMALLOC
#ifdef GKMSPACE
		mspace_free(gk_mspace, *ptr1);
#else
		dlfree(*ptr1);
#endif
#else
		free(*ptr1);
#endif
	*ptr1 = NULL;

	va_start(plist, ptr1);

	while ((ptr = va_arg(plist, void **) ) != LTERM ) {
		if (*ptr != NULL )
#ifdef USE_DLMALLOC
#ifdef GKMSPACE
			mspace_free(gk_mspace, *ptr);
#else
			dlfree(*ptr);
#endif
#else
			free(*ptr);
#endif
		*ptr = NULL;
	}

	va_end(plist);
}
