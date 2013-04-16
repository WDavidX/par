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

#define wcint(a,n,offset,id)\
	do{ int local_k;\
	fprintf(stderr,"ID:%2d \tLine %d, ArrayInt: %s \t Number:%d \t offset %d\n",(id),__LINE__, #a,n,offset);\
		for (local_k=0;local_k<n;++local_k)\
			fprintf(stderr,"%s[%d] = %d\n",#a,(local_k+offset),(a[local_k]));\
	}while(0)

#define wcdouble(a,n,offset,id))\
	do{ int local_k;\
	fprintf(stderr,"ID:%2d \tLine %d, ArrayDouble: %s \t Number:%d \t offset %d\n",(id),__LINE__, #a,n,offset);\
		for (local_k=0;local_k<n;++local_k)\
			fprintf(stderr,"%s[%d] = %5.3lf\n",#a,(local_k+offset),(a[local_k]));\
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
	/* ********** Timer Info ********** */
	double t10; // reading files
	double t11; // send matrix all over
	double t0;  // global timer
	double t1, t2, t3;
	/* ********** Per Node Basic Info ********** */

	char proc_name[MPI_MAX_PROCESSOR_NAME];
	int namelength;
	/* ********** File Info ********** */
	char *fmat, *fvec, *fout;
	/* ********** Key Info ********** */
	int np, id;
	int nnz, nrow;

	/* ********** Row start Info ********** */
	int *blknnzst;
	int *blkrowst;

	/* ********** Local block i,j,val, vec********** */
	double *localval;
	double *localvec;
	int *locali;
	int *localj;
	int localnrow;
	int localnnz;
} par_t;

typedef struct {
	int nrow, nnz, blknrow, blknnz;
	double *tmpb, *b;				//tmpb: tmp save b; before inspection;
	int bct, tmpbct, browstart;		//number of element in tmpb
} mat_t;

/* ********** Global Varibles********** */
par_t par;
mat_t mat;
int all[2] = { -1, -1 };

int Step1root(par_t *par, mat_t *mat);
int Step1other(par_t *par, mat_t *mat);

/*
// int MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)

 int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)

 int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm )


 */
int main(int argc, char *argv[]) {
	/* ******************** Step 0: Get Host Names and IDs ******************** */
	MPI_Init(&argc, &argv); /* starts MPI */
	par.t0 = par.t1 = par.t2 = par.t3 = par.t10 = par.t11 = 0;
	Tstart(par.t0);
	MPI_Comm_rank(MPI_COMM_WORLD, &par.id); /* get current process id */
	MPI_Comm_size(MPI_COMM_WORLD, &par.np); /* get number of processes */
	MPI_Get_processor_name(par.proc_name, &par.namelength);

	if (par.id == 0) {
		par.fmat = argv[1];
		par.fvec = argv[2];
		par.fout = NULL;
	} else {
		par.fmat = par.fvec = par.fout = NULL;
	};

	DPRINTF("Proc %d of %d, %s, fmat=%s, fvec=%s\n", par.id, par.np,
			par.proc_name, par.fmat, par.fvec);
	MPI_Barrier(MPI_COMM_WORLD);

	/* ******************** Step 1: Distribute the data ******************** */
	if (par.id == 0) {
		Step1root(&par, &mat);
	} else {
		Step1other(&par, &mat);
	}

	/* ******************** Step 2: Analyze non-zero entry******************** */

	/* ******************** Step 3: Distribute the data ******************** */

	/* ******************** Step 4: Distribute the data ******************** */

	/* ******************** Step 5: Distribute the data ******************** */

	/* ******************** Step 6: Distribute the data ******************** */

	if (par.id == 0) {
		Tstop(par.t0);
		printf("======================================================\n");
		printf("Overall time         t0   (sec):\t\t%.8lf\n", par.t0);
		printf("Total time taken     t1   (sec):\t\t%.8lf\n", par.t1);
		printf("Time taken (step5&6) t2   (sec):\t\t%.8lf\n", par.t2);
		printf("Time taken (fileio) t10   (sec):\t\t%.8lf\n", par.t10);
		printf("======================================================\n");
	}

	MPI_Finalize();

	return 0;
}

int Step1root(par_t *par, mat_t *mat) {
	int k; // universal iterator
	int bk; // interator for each block
	/* ********** Step 1.1: Broadcast NNZ / NROW, file io ready ********** */
	Tstart(par->t10);
	par->nnz = TextLines(par->fmat);
	par->nrow = TextLines(par->fvec);
	Tstop(par->t10);
	all[0] = par->nnz;
	all[1] = par->nrow;
	MPI_Bcast(&all, 2, MPI_INT, par->id, MPI_COMM_WORLD);

	par->fout = (char*) malloc(sizeof(char) * 256);
	sprintf(par->fout, "./o%d.vec", par->nrow);
	FILE *fA; // For matrix file
	FILE *fb; // For vector file
	if ((fA = fopen(par->fmat, "r")) == NULL
			|| (fb = fopen(par->fvec, "r")) == NULL ) {
		fprintf(stderr, "Fail to open files\n");
		exit(1);
	}

	/* ********** Step 1.2: Calculate blkrowst and allocate local b********** */
	wcalloc(par->blknnzst, int, par->np + 1, "Alloc blknnzst");
	wcalloc(par->blkrowst, int, par->np + 1, "Alloc blkrowst");
	for (bk = 0; bk < par->np; ++bk) {
		par->blkrowst[bk] = (par->nrow / par->np) * bk;
	}
	par->blkrowst[par->np] = par->nrow;
	par->localnrow = par->blkrowst[par->id + 1] - par->blkrowst[par->id];
	wcalloc(par->localvec, double, par->localnrow, "local vector val");
//	DPRINTF("ID=%d, localnrow=%d\n",par->id,par->localnrow);
	double *allvecval;

	wcalloc(allvecval, double, par->nrow, "all vector val");
	for (k = 0; k < par->nrow; ++k) {
		if ((fscanf(fb, "%lf", &allvecval[k]) == 0))
			fprintf(stderr, "Error reading vec b[%d]\n", k);
//		printf("vecb[%d]= %.4lf\n", k, allvecval[k]);
	}

	/* ********** Step 1.3: Send vector b ********** */
	//local copy: void * memcpy ( void * destination, const void * source, size_t num );
	memcpy(par->localvec, &allvecval[par->id], sizeof(double) * par->localnrow);
//	wcdouble(par->localvec,par->localnrow,0);
	for (bk = 1; bk < par->np; ++bk) {
		MPI_Send(&allvecval[bk * (par->nrow / par->np)],
				par->blkrowst[bk + 1] - par->blkrowst[bk], MPI_DOUBLE, bk, 0,
				MPI_COMM_WORLD);
	}
	wcfree(allvecval, "Free all vector values");

	/* ********** Step 1.4root: Read matrix file and send blknnzst table ********** */
	int *alli, *allj;
	double *allval;
	char *blktouched;
	wccalloc(blktouched, char, par->np + 1, "get first array");
	wcalloc(alli, int, par->nnz, "all matrix i");
	wcalloc(allj, int, par->nnz, "all matrix j");
	wcalloc(allval, double, par->nnz, "all matrix val");

	int blkcounter = 0;
	for (k = 0; k < par->nnz; ++k) {
		if (fscanf(fA, "%d %d %lf", &alli[k], &allj[k], &allval[k]) == 0) {
			fprintf(stderr, "Error reading matrix entry %d\n", k);
		}
		while (blktouched[blkcounter] == 0
				&& alli[k] >= (par->blkrowst[blkcounter])) {
			blktouched[blkcounter] = 1;
			par->blknnzst[blkcounter] = k;
			blkcounter++;
		}
	}
	par->blknnzst[blkcounter] = par->nnz;
	MPI_Bcast(par->blknnzst, par->np + 1, MPI_INT, par->id, MPI_COMM_WORLD);
	par->localnnz=par->blknnzst[par->id+1]-par->blknnzst[par->id];
	/* ********** Step 1.4: Send to local i,j,val of matrix  ********** */
	for (bk = 1; bk < par->np; ++bk) {
//		 int MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
//		MPI_Send(&alli[bk],par->blknnzst[bk+1]-par->blknnzst[bk],MPI_INT,bk,0,MPI_COMM_WORLD);
	}
	return -1;
}



int Step1other(par_t *par, mat_t *mat) {
	int bk; // block iterator
	/* ********** Step 1.2: Broad NNZ and NROW ********** */
	MPI_Status status;
	MPI_Bcast(&all, 2, MPI_INT, 0, MPI_COMM_WORLD);
	par->nnz = all[0];
	par->nrow = all[1];
	DPRINTF("NNZ=%d, NROW=%d, id=%d\n",all[0],all[1],par->id);

	/* ********** Step 1.2: Calculate blkrowst and allocate local b********** */
	wcalloc(par->blknnzst, int, par->np + 1, "Alloc blknnzst");
	wcalloc(par->blkrowst, int, par->np + 1, "Alloc blkrowst");
	for (bk = 0; bk < par->np; ++bk) {
		par->blkrowst[bk] = (par->nrow / par->np) * bk;
	}
	par->blkrowst[par->np] = par->nrow;
	par->localnrow = par->blkrowst[par->id + 1] - par->blkrowst[par->id];
	wcalloc(par->localvec, double, par->localnrow, "local vector val");
//	DPRINTF("ID=%d, localnrow=%d\n",par->id,par->localnrow);

	/* ********** Step 1.3: Recv vector b ********** */
	MPI_Recv(par->localvec, par->localnrow, MPI_DOUBLE, 0, MPI_ANY_TAG,
			MPI_COMM_WORLD, &status);

	/* ********** Step 1.4other: Recv blknnzst table ********** */
	MPI_Bcast(par->blknnzst, par->np + 1, MPI_INT, 0, MPI_COMM_WORLD);
	par->localnnz=par->blknnzst[par->id+1]-par->blknnzst[par->id];
	wcint(par->blknnzst, par->np + 1, 0, par->id);

	/* ********** Step 1.4: Recv to local i,j,val of matrix  ********** */
//	MPI_Recv(&par->blknnzst[bk],par->blknnzst[bk+1]-par->blknnzst[bk],MPI_INT,bk,0,MPI_COMM_WORLD);
	return -1;
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

