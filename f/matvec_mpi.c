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

/* Helper functions */
int TextLines(char *namestr);
#define LTERM                   (void **) 0     /* List terminator for GKfree() */
#define PATHPREFIX "./"
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
	fprintf(stdout,"ID:%2d \tLine %d, ArrayInt: %s   Number:%d   offset %d\n",(id),__LINE__, #a,n,offset);\
		for (local_k=0;local_k<n;++local_k)\
			fprintf(stdout,"ID %d %s[%d] = %d\n",(id),#a,(local_k+offset),(a[local_k]));\
	}while(0)

#define wcdouble(a,n,offset,id)\
	do{ int local_k;\
	fprintf(stdout,"ID:%2d \tLine %d, Double: %s  Number:%d  offset %d\n",(id),__LINE__, #a,n,offset);\
		for (local_k=0;local_k<n;++local_k)\
			fprintf(stdout,"ID %d %s[%d] = %5.2lf\n",(id),#a,(local_k+offset),(a[local_k]));\
	}while(0)

#ifdef DEBUG
#define wcary(a,b,n,id)\
	do{ int local_k;\
		for (local_k=0;local_k<n;++local_k){\
			if (a[local_k]!=b[local_k]){\
			fprintf(stdout,"Unequal entry ID %d \t %s[%d] = %d; %s[%d] = %d\n",(id),#a,local_k,a[local_k],#b,local_k,b[local_k]);\
			exit(1);}}\
	}while(0)
#else
#define wcary(a,b,n,id) do{} while(0)
#endif

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
#define PASSHERE(x) do{fprintf(stderr,"ID: %d   Pass line %d    %s\n",(par.id),__LINE__,x);}while(0)
#else
#define PASSHERE(x) do{}while(0)
#endif

#ifdef DEBUG
#define IDPASS(id,x) do{fprintf(stdout,"ID: %d   Pass line %d    %s\n",(id),__LINE__,x);}while(0)
#else
#define IDPASS(id,x) do{}while(0)
#endif

#ifdef DEBUG
#define Bar(id) do{MPI_Barrier(MPI_COMM_WORLD);fprintf(stdout,"======>ID: %d   Reach barrier line %d\n",(id),__LINE__);MPI_Barrier(MPI_COMM_WORLD);}while(0)
#else
#define Bar(id) do{}while(0)
#endif

typedef struct {
	/* ********** Timer Info ********** */
	double t10; // reading files
	double t11; // timer to distribute the data
	double t12; // timer to write data into file
	double t13; // actual local computation
	double t0;  // global timer
	double t1;  // report step 2~6 time
	double t2;  // report step 5&6 time
	double t3;
	/* ********** Per Node Basic Info ********** */
	int PID;
	char proc_name[MPI_MAX_PROCESSOR_NAME];
	int namelength;
	/* ********** File Info ********** */
	char *fmat, *fvec, *fout, *foutdir;
	/* ********** Key Info ********** */
	int np, id;
	int nnz, nrow;

	/* ********** Row start Info ********** */
	int *blknnzst;
	int *blkrowst;

	/* ********** Local block i,j,val, vec********** */
	double *localvec;
	int *locali;
	int *localj;
	double *localval;
	int localnrow;
	int localnnz;

	/* ********** Final output ********** */
	double *finalout;
} par_t;

typedef struct {
	/* ********** Overall info ********** */
	int nrow, ncol, nnz;
	int matrowst, matrownum;
	/* ********** Compressed Sparse Row CSR ********** */
	double *colval;
	int *colind;
	int *rowptr;
	/* ********** Compressed Sparse Column CSC ********** */
	double *rowval;
	int *rowind;
	int *colptr;
	/* ********** All needed vector entries ********** */
	double *b;
	int *marked;     // mark the entries that will be used
	int *needbidx;   // store the index of b's entry that is used at least once
	int *nonlocalbst;  // nonlocal b start index in needbidx size of np+1
	int bsize;   // how many vector entries need from all blocks, sizrof np;
	/* ********** Output vector ********** */
	double *vout; // should have size of localnrows
//may need more
} mat_t;

/* ********** Global Varibles********** */
par_t par;
mat_t mat;
int all[2] = { -1, -1 };

int Step1root(par_t *par);
int Step1other(par_t *par);

void Init(par_t *par, mat_t *mat);
void CreateCSR(par_t *par, mat_t *mat);
void IndexCSC(par_t *par, mat_t *mat);
void DeterminNonlocalb(par_t *par, mat_t *mat);
void TransferVector(par_t *par, mat_t *mat);
void ComputeOutput(par_t *par, mat_t *mat);
void GatherOutput(par_t *par, mat_t *mat);
void WriteOutput(par_t *par, mat_t *mat);
void FinalFreeMemory(par_t *par, mat_t *mat);

int main(int argc, char *argv[]) {
	/* ******************** Step 0: Get Host Names and IDs ******************** */
	MPI_Init(&argc, &argv); /* starts MPI */
	par.t0 = 0;
	Tstart(par.t0);
	MPI_Comm_rank(MPI_COMM_WORLD, &par.id); /* get current process id */
	MPI_Comm_size(MPI_COMM_WORLD, &par.np); /* get number of processes */
	MPI_Get_processor_name(par.proc_name, &par.namelength);
	Init(&par, &mat);
	/* ******************** Step 0: Set some init values ******************** */
	if (par.id == 0) {
		if (argc == 4) {
			par.foutdir = argv[3];
		} else {
			if (argc > 4 || argc < 2) {
				fprintf(stderr,
						"Wrong number of argument. ./matrix.ij ./vector.vec");
				exit(1);
			} else
				par.foutdir = NULL;
		}
		par.fmat = argv[1];
		par.fvec = argv[2];
		par.fout = NULL;
	} else {
		par.fmat = par.fvec = par.fout = par.foutdir = NULL;
	};

	if (par.id == 0) {
		printf("PID %6d: Proc %2d of %d, %s\n", par.PID, par.id, par.np,
				par.proc_name);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	/* ******************** Step 1: Distribute the data ******************** */
	if (par.id == 0) {
		Step1root(&par);
	} else {
		Step1other(&par);
	}
	CreateCSR(&par, &mat); // here, all nodes have local mat and vec in CSR form
	MPI_Barrier(MPI_COMM_WORLD);
	/* ******************** Step 2: Analyze non-zero entry******************** */
	IndexCSC(&par, &mat);
	/* ******************** Step 3: Analyze non-zero entry******************** */
	DeterminNonlocalb(&par, &mat);

	/* ******************** Step 4: Figure out which process needs which ******************** */
	/* ******************** Step 5: Actual transfer the data ******************** */
	Tstart(par.t2);
	Tstart(par.t11);
	TransferVector(&par, &mat);
	Tstop(par.t11);
	/* ******************** Step 6: Computing local product ******************** */
	Tstart(par.t13);
	ComputeOutput(&par, &mat);
	Tstop(par.t13);
	Tstop(par.t2);
	/* ******************** Step 7: Gather Data ******************** */
	GatherOutput(&par, &mat);
	if (par.id == 0) {
		WriteOutput(&par, &mat);
		Tstop(par.t0);
		printf("\n======================================================\n");
		printf("Overall time         t0   (sec):\t%.8lf\n", par.t0);
		printf("Total time taken     t1   (sec):$\t%.8lf\n", par.t1);
		printf("Time taken (step5&6) t2   (sec):$\t%.8lf\n", par.t2);
		printf("Time taken (fileio) t10   (sec):\t%.8lf\n", par.t10);
		printf("Transfer needed vec t11   (sec):\t%.8lf\n", par.t11);
		printf("Local computation   t13   (sec):\t%.8lf\n", par.t13);
		printf("Write output file   t12   (sec):\t%.8lf\n", par.t12);
		printf("======================================================\n");
	}
	FinalFreeMemory(&par, &mat);
	MPI_Finalize();

	return 0;
}

void Init(par_t *par, mat_t *mat) {
	par->PID = getpid();
	par->t1 = par->t2 = par->t3 = 0;
	par->t10 = par->t11 = par->t12 = par->t13 = 0;
	mat->nnz = -1;
	mat->nrow = -1;
	mat->colind = mat->rowind = mat->rowptr = mat->colptr = NULL;
	mat->colval = mat->rowval = NULL;
	par->finalout = NULL;
	mat->colval = NULL;
	mat->colind = NULL;
	mat->rowptr = NULL;
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
	printf("Number of lines in %20s \t = %d\n", namestr, nlines);
	return nlines;
}

int Step1root(par_t *par) {
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
	FILE *fA; // For matrix file
	FILE *fb; // For vector file

	if (par->foutdir == NULL )
		sprintf(par->fout, "%so%d.vec", PATHPREFIX, par->nrow);
	else
		sprintf(par->fout, "%so%d.vec", par->foutdir, par->nrow);

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
	double *allvecval;
	wcalloc(allvecval, double, par->nrow, "all vector val");
	Tstart(par->t10);
	for (k = 0; k < par->nrow; ++k) {
		if ((fscanf(fb, "%lf", &allvecval[k]) == 0))
			fprintf(stderr, "Error reading vec b[%d]\n", k);
	}
	Tstop(par->t10);

	/* ********** Step 1.3: Send vector b ********** */
	memcpy(par->localvec, &allvecval[par->id], sizeof(double) * par->localnrow);
	for (bk = 1; bk < par->np; ++bk) {
		MPI_Send(&allvecval[bk * (par->nrow / par->np)],
				par->blkrowst[bk + 1] - par->blkrowst[bk], MPI_DOUBLE, bk, 0,
				MPI_COMM_WORLD);
	}

	/* ********** Step 1.4root: Read matrix file and send blknnzst table ********** */
	int *alli, *allj;
	double *allval;
	char *blktouched;
	wccalloc(blktouched, char, par->np + 1, "get first array");
	wcalloc(alli, int, par->nnz, "all matrix i");
	wcalloc(allj, int, par->nnz, "all matrix j");
	wcalloc(allval, double, par->nnz, "all matrix val");
	int blkcounter = 0;
	Tstart(par->t10);
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
	Tstop(par->t10);
	par->blknnzst[blkcounter] = par->nnz;
	MPI_Bcast(par->blknnzst, par->np + 1, MPI_INT, par->id, MPI_COMM_WORLD);
	par->localnnz = par->blknnzst[par->id + 1] - par->blknnzst[par->id];
	wcalloc(par->locali, int, par->localnnz, "Recv node local i");
	wcalloc(par->localj, int, par->localnnz, "Recv node local j");
	wcalloc(par->localval, double, par->localnnz, "Recv node local val");

	/* ********** Step 1.4: Send to local i,j,val of matrix  ********** */
	for (bk = 1; bk < (par->np); ++bk) {
		MPI_Send(&alli[par->blknnzst[bk]],
				par->blknnzst[bk + 1] - par->blknnzst[bk], MPI_INT, bk, 0,
				MPI_COMM_WORLD);
		MPI_Send(&allj[par->blknnzst[bk]],
				par->blknnzst[bk + 1] - par->blknnzst[bk], MPI_INT, bk, 0,
				MPI_COMM_WORLD);
		MPI_Send(&allval[par->blknnzst[bk]],
				par->blknnzst[bk + 1] - par->blknnzst[bk], MPI_DOUBLE, bk, 0,
				MPI_COMM_WORLD);
	}
	memcpy(par->locali, alli, sizeof(int) * par->localnnz);
	memcpy(par->localj, allj, sizeof(int) * par->localnnz);
	memcpy(par->localval, allval, sizeof(double) * par->localnnz);

	wcfree(alli, "Free all i");
	wcfree(allj, "Free all j");
	wcfree(allval, "Free all val");
	wcfree(allvecval, "Free all vec");
	wcfree(blktouched, "blktouched");
	fclose(fA);
	fclose(fb);
	return -1;
}

int Step1other(par_t *par) {
	int bk; // block iterator
	/* ********** Step 1.2: Broad NNZ and NROW ********** */
	MPI_Status status;
	MPI_Bcast(&all, 2, MPI_INT, 0, MPI_COMM_WORLD);
	par->nnz = all[0];
	par->nrow = all[1];

	/* ********** Step 1.2: Calculate blkrowst and allocate local b********** */
	wcalloc(par->blknnzst, int, par->np + 1, "Alloc blknnzst");
	wcalloc(par->blkrowst, int, par->np + 1, "Alloc blkrowst");
	for (bk = 0; bk < par->np; ++bk) {
		par->blkrowst[bk] = (par->nrow / par->np) * bk;
	}
	par->blkrowst[par->np] = par->nrow;
	par->localnrow = par->blkrowst[par->id + 1] - par->blkrowst[par->id];
	wcalloc(par->localvec, double, par->localnrow, "local vector val");

	/* ********** Step 1.3: Recv vector b ********** */
	MPI_Recv(par->localvec, par->localnrow, MPI_DOUBLE, 0, MPI_ANY_TAG,
			MPI_COMM_WORLD, &status);

	/* ********** Step 1.4other: Recv blknnzst table ********** */
	MPI_Bcast(par->blknnzst, par->np + 1, MPI_INT, 0, MPI_COMM_WORLD);
	par->localnnz = par->blknnzst[par->id + 1] - par->blknnzst[par->id];
	wcalloc(par->locali, int, par->localnnz, "Recv node local i");
	wcalloc(par->localj, int, par->localnnz, "Recv node local j");
	wcalloc(par->localval, double, par->localnnz, "Recv node local val");
	/* ********** Step 1.4: Recv to local i,j,val of matrix  ********** */
	MPI_Recv(par->locali, par->localnnz, MPI_INT, MPI_ANY_TAG, 0,
			MPI_COMM_WORLD, &status);
	MPI_Recv(par->localj, par->localnnz, MPI_INT, MPI_ANY_TAG, 0,
			MPI_COMM_WORLD, &status);
	MPI_Recv(par->localval, par->localnnz, MPI_DOUBLE, MPI_ANY_TAG, 0,
			MPI_COMM_WORLD, &status);
//	wcint(par->locali,par->localnnz,0,par->id);
//	wcint(par->localj,par->localnnz,0,par->id);
//	wcdouble(par->localval,par->localnnz,0,par->id);
	return -1;
}

void CreateCSR(par_t *par, mat_t *mat) {
	int k;
	mat->nrow = par->localnrow;
	mat->nnz = par->localnnz;
	mat->ncol = par->nrow;

	/* ********** Mark whether an entry in b has been used ********** */
	wccalloc(mat->marked, int, mat->ncol, "Zero init mat marked"); // Init to zero

	/* ********** Allocate Space for CSR ********** */
	wcalloc(mat->colind, int, mat->nnz, "Mat alloc colind");
	wcalloc(mat->colval, double, mat->nnz, "Mat alloc colval");
	wcalloc(mat->rowptr, int, mat->nrow + 1, "Mat alloc rowptr");
	/* ********** Allocate Space for CSS ********** */
	wcalloc(mat->rowind, int, mat->nnz, "Mat alloc rowind");
	wcalloc(mat->rowval, double, mat->nnz, "Mat alloc rowval");
	wcalloc(mat->colptr, int, mat->ncol + 1, "Mat alloc colptr");
	int currentrow = 0;
	mat->rowptr[currentrow] = 0;
	for (k = 0; k < mat->nnz; ++k) {
		while ((par->locali[k] - par->blkrowst[par->id]) > currentrow) {
			++currentrow;
			mat->rowptr[currentrow] = k;
		}
		mat->colind[k] = par->localj[k];
		mat->colval[k] = par->localval[k];
	}
	mat->rowptr[mat->nrow] = mat->nnz;

}

void IndexCSC(par_t *par, mat_t *mat) {

	int k;
	if (((mat->colind) && (mat->colval) && (mat->rowptr)) == 0) {
		fprintf(stderr, "The CSR form is not ready yet");
	}
	for (k = 0; k < mat->nnz; ++k) {
		(mat->marked)[mat->colind[k]]++;
	}

	mat->bsize = 0;
	int currentblk = 0;
	for (k = 0; k < mat->ncol; ++k) {
		/* ********** Calculate the exclusive prefix sum ********** */
		if (k == 0)
			mat->colptr[k] = 0;
		else
			mat->colptr[k] = mat->colptr[k - 1] + mat->marked[k - 1];
		/* ********** Calculate the exclusive prefix sum ********** */
		if (mat->marked[k] != 0) {
			mat->bsize++;
			while (k >= par->blkrowst[currentblk + 1]) {
				currentblk++;
			}
		}
	}
	mat->colptr[k] = mat->nnz;
	int *colacc;
	wccalloc(colacc, int, mat->ncol, "colptr accummulator");

	int r, cpt, c, nnzcounter = 0;
	for (r = 0; r < mat->nrow; ++r) {
		for (cpt = mat->rowptr[r]; cpt < mat->rowptr[r + 1]; ++cpt) {
			nnzcounter++;
			c = mat->colind[cpt];
			mat->rowval[mat->colptr[c] + colacc[c]] = mat->colval[cpt];
			mat->rowind[mat->colptr[c] + colacc[c]] = r;
			colacc[c]++;
		}
	}
	wcfree(colacc, "accumulator for each column");
}

void DeterminNonlocalb(par_t *par, mat_t *mat) {
	int k;
	/* ********** How many entries are needed from other block ********** */
	wccalloc(mat->nonlocalbst, int, par->np + 1, "Mat blkbsize");
	/* ********** Allocate space for of index of all needed B */
	wcalloc(mat->needbidx, int, mat->bsize, ""); //in between

	int nowptrb = 0;		// index of needbidx array
	int blkcounter = 0;
	/* ********** Iterator through all columns ********** */
	for (k = 0; k < mat->ncol; ++k) {
		/* ********** block starts here ********** */
		if (k == par->blkrowst[blkcounter]) {
			mat->nonlocalbst[blkcounter] = nowptrb;
			++blkcounter;
		}
		/* ********** Put needed b in continous memories ********** */
		if (mat->marked[k] != 0) {
			mat->needbidx[nowptrb] = k;
			++nowptrb;
		}
	}
	mat->nonlocalbst[blkcounter] = mat->bsize;
}

void TransferVector(par_t *par, mat_t *mat) {
	int k;
	MPI_Status status;
	/* ********** Allocate space for all needed B in space O(bsize)=O(n/p) ********** */
	wcalloc(mat->b, double, mat->bsize, " alloc space for all needed b");

	/* ********** Doing local copy ********** */
	for (k = mat->nonlocalbst[par->id]; k < mat->nonlocalbst[par->id + 1];
			++k) {
		mat->b[k] = par->localvec[mat->needbidx[k] - par->blkrowst[par->id]];
	}

	/* ********** Doing cyclic remote copy ********** */
	int idstep;
	int outdst, insrc;
	int sendbufsize = 0, recvbufsize = 0;
	int *recvidxbuf = NULL;
	double *sendvalbuf = NULL;
	for (idstep = 1; idstep < par->np; ++idstep) {

		outdst = ((par->id) + idstep) % (par->np);		// send get request to
		insrc = ((par->id) + par->np - idstep) % (par->np); // recv get request from
		/* ********** This is a must send to outdst, received from insrc ********** */
		/* ********** Send number of indices ********** */
		sendbufsize = mat->nonlocalbst[outdst + 1] - mat->nonlocalbst[outdst];
		MPI_Send(&sendbufsize, 1, MPI_INT, outdst, 0, MPI_COMM_WORLD);
		/* ********** Receive number of indices ********** */
		MPI_Recv(&recvbufsize, 1, MPI_INT, insrc, MPI_ANY_TAG, MPI_COMM_WORLD,
				&status);

		/* ********** Receive number of indices ********** */
		if (recvbufsize != 0) {
			wcalloc(recvidxbuf, int, recvbufsize,
					"recvbuf get requested index");
			wcalloc(sendvalbuf, double, recvbufsize,
					"send out requested values");
		}

		/* ********** Send indices list ********** */
		if (sendbufsize != 0)
			MPI_Send(&mat->needbidx[mat->nonlocalbst[outdst]], sendbufsize,
					MPI_INT, outdst, 0, MPI_COMM_WORLD);

		/* ********** Receive indices list ********** */
		if (recvbufsize != 0)
			MPI_Recv(recvidxbuf, recvbufsize, MPI_INT, insrc, 0, MPI_COMM_WORLD,
					&status);

		/* ********** send to insrc, received from outdst ********** */
		/* ********** Prepare value list for insrc********** */

		for (k = 0; k < recvbufsize; ++k) {
			sendvalbuf[k] =
					par->localvec[recvidxbuf[k] - par->blkrowst[par->id]];
		}

		/* ********** Send value list ********** */
		MPI_Send(sendvalbuf, recvbufsize, MPI_DOUBLE, insrc, 0, MPI_COMM_WORLD);

		/* ********** Receive value list ********** */
		MPI_Recv(&mat->b[mat->nonlocalbst[outdst]], sendbufsize, MPI_DOUBLE,
				outdst, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		/* ********** Clear the send and received buffers if not empty ********** */
		if (recvbufsize != 0) {
			wcfree(recvidxbuf, "recv index buffer freed");
			wcfree(sendvalbuf, "recv index buffer freed");
		}
	}
}

void ComputeOutput(par_t *par, mat_t *mat) {
	int k;	// column iterator
	/* ********** Allocate space for local results ********** */
	wccalloc(mat->vout, double, mat->nrow, "block output");
	int r, c;	// actual local row and column
	for (k = 0; k < mat->bsize; ++k) {
		c = mat->needbidx[k];
		for (r = mat->colptr[c]; r < mat->colptr[c + 1]; ++r) {
			mat->vout[mat->rowind[r]] += mat->b[k] * mat->rowval[r];
		}
	}
}

void GatherOutput(par_t *par, mat_t *mat) {
	int k;
	MPI_Status status;
	if (par->id == 0) {
		wcalloc(par->finalout, double, par->nrow, " final out");
		memcpy(par->finalout, mat->vout,
				sizeof(double) * (par->blkrowst[1] - par->blkrowst[0]));
		for (k = 1; k < par->np; ++k) {
			MPI_Recv(&par->finalout[par->blkrowst[k]],
					(par->blkrowst[k + 1] - par->blkrowst[k]), MPI_DOUBLE, k,
					MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		}
	} else {
		/* ********* p2p send needed if nrow is not multiple of np   ********* */
		MPI_Send(mat->vout, mat->nrow, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
}

void WriteOutput(par_t *par, mat_t *mat) {
	int k;
	FILE *f;
	printf("Output file name: %s", par->fout);
	if (par->finalout == NULL ) {
		fprintf(stderr, "Final out vector is NULL\n");
	}
	if ((f = fopen(par->fout, "w")) == NULL ) {
		fprintf(stderr, "Fail to open output file: %s", par->fout);
		return;
	}
	Tstart(par->t12);
	for (k = 0; k < par->nrow; ++k) {
		fprintf(f, "%.15lf\n", par->finalout[k]);
	}
	fclose(f);
	Tstop(par->t12);
}

void FinalFreeMemory(par_t *par, mat_t *mat) {
	wcfree(par->locali, "");
	wcfree(par->localj, "");
	wcfree(par->localval, "");
	wcfree(par->localvec, "");
	wcfree(par->blknnzst, "");
	wcfree(par->blkrowst, "");
	if (par->id == 0) {
		wcfree(par->finalout, "");
		wcfree(par->fout, "");
	}

	wcfree(mat->rowind, "");
	wcfree(mat->colind, "");
	wcfree(mat->rowval, "");
	wcfree(mat->colval, "");
	wcfree(mat->rowptr, "");
	wcfree(mat->colptr, "");
	wcfree(mat->b, "");
	wcfree(mat->needbidx, "");
	wcfree(mat->nonlocalbst, "");
	wcfree(mat->vout, "");
}

//1. One process reads the file and distributes the data to the other processes using a 1D decomposition along the rows. The decomposition for A and b should be the same. 
//2. Each process analyzes the non-zero elements of A that it stores and determines which non-local element of b it will need in order to compute its entries of x.
//3. Each process determines which processors stores the non-local b elements.
//4. All the processes are communicating to figure out which process needs to send what data to the other processes.
//5. The processes perform the actual transfers of the elements of b.
//6. Each process now has all the elements of b that it needs (and nothing more) and proceeds to compute the elements of the x vector that it is responsible for.
//7. The results are gathered to a single process, which writes them to the disk.
