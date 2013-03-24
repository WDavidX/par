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
	int num_procs, myid, name_len, idx, power2, minidx;
	char *proc_name;
} params_t;

#include <mpi.h>

#define Tclear(tmr) (tmr = 0.0)
#define Tstart(tmr) (tmr -= MPI_Wtime())
#define Tstop(tmr)  (tmr += MPI_Wtime())
#define Tget(tmr)   (tmr)

//#define RAND_RANGE INT_MAX
#define RAND_RANGE 10

#define MYRANDSEED (time(NULL))
//#define MYRANDSEED (1)

#define EXTRA 0
#define MIN(a,b) (((a)<(b))?(a):(b))

params_t par;

void Init(int argc, char *argv[], params_t *par);
int myMPI_Scan(const void *sendbuf, void *recvbuf, int count,
		MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
void cleanup(params_t *par);

int SameOutput(int* a, int* b, int num, params_t *par);

void PrintArray(params_t *par);
void PrintArrayC(params_t *par);
void PrintArrayA(params_t *par);

void InsertNewLine(params_t *par) {
	MPI_Barrier(MPI_COMM_WORLD );
	if (par->myid == 0) {
		printf("\n");
	}
	MPI_Barrier(MPI_COMM_WORLD );
}

int main(int argc, char *argv[]) {

	MPI_Init(&argc, &argv);
	int num_procs, myid, name_len, k;
	double timer_scan, timer_myscan;
	char proc_name[MPI_MAX_PROCESSOR_NAME];
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Get_processor_name(proc_name, &name_len);
	printf("Proc %d/%d,%6s\n", myid, num_procs, proc_name);
	par.num_procs = num_procs;
	par.myid = myid;
	par.name_len = name_len;
	par.proc_name = (char*) &proc_name;
	MPI_Barrier(MPI_COMM_WORLD );
	Init(argc, argv, &par);

	/* ********** Run MPI_Scan ********** */
	MPI_Barrier(MPI_COMM_WORLD );
	Tclear(par.timer_1);
	Tstart(par.timer_1);
	MPI_Scan(par.c, par.a, par.n, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
	Tstop(par.timer_1);
	timer_scan = par.timer_1;
	MPI_Barrier(MPI_COMM_WORLD );

	for (k = 0; k < EXTRA; ++k) {
		MPI_Barrier(MPI_COMM_WORLD );
		Tclear(par.timer_1);
		Tstart(par.timer_1);
		MPI_Scan(par.c, par.a, par.n, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
		Tstop(par.timer_1);
		timer_scan = MIN(par.timer_1,timer_scan);
		MPI_Barrier(MPI_COMM_WORLD );
	}
	par.timer_1 = timer_scan;

	/* ********** Run myMPI_Scan ********** */
	MPI_Barrier(MPI_COMM_WORLD );
	Tclear(par.timer_2);
	Tstart(par.timer_2);
	myMPI_Scan(par.c, par.b, par.n, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
	Tstop(par.timer_2);
	timer_myscan = par.timer_2;
	MPI_Barrier(MPI_COMM_WORLD );

	for (k = 0; k < EXTRA; ++k) {
		MPI_Barrier(MPI_COMM_WORLD );
		Tclear(par.timer_2);
		Tstart(par.timer_2);
		myMPI_Scan(par.c, par.b, par.n, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
		Tstop(par.timer_2);
		timer_myscan = MIN(par.timer_2,timer_myscan);
		MPI_Barrier(MPI_COMM_WORLD );
	}
	par.timer_2 = timer_myscan;

	/* ********** Check both scan have idential output ********** */
	SameOutput(par.a, par.b, par.n, &par);


	/* ********** Print time at node with largest rank ********** */
	MPI_Barrier(MPI_COMM_WORLD );
	if (par.myid == par.num_procs - 1) {
		printf("MPI_myscan:   np=%2d   \tn=%7d\t host %s\n",par.num_procs,par.n,par.proc_name);
		printf("  OMP_SCAN timer1 (sec) on %d runs: \t %.8lf", 1 + EXTRA,
				par.timer_1);
		printf("\nmyOMP_SCAN timer2 (sec) on %d runs: \t %.8lf", 1 + EXTRA,
				par.timer_2);
	}
	MPI_Barrier(MPI_COMM_WORLD );

	cleanup(&par);
	MPI_Finalize();
	return 0;
}

void PrintArray(params_t *par) {
//	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, par->myid, 0, win);
	int prc_num;
	for (prc_num = 0; prc_num < par->num_procs; ++prc_num) {
		MPI_Barrier(MPI_COMM_WORLD );
		if (par->myid == prc_num) {
			printf("\n");
			printf("Proc %d/%d,%6s A ", par->myid, par->num_procs,
					par->proc_name);
			int k;
			for (k = 0; k < par->n; ++k) {
				printf("%3d ", par->a[k]);
			}
			printf("\n");
			printf("Proc %d/%d,%6s B ", par->myid, par->num_procs,
					par->proc_name);
			for (k = 0; k < par->n; ++k) {
				printf("%3d ", par->b[k]);
			}
			printf("\n");
			printf("Proc %d/%d,%6s C ", par->myid, par->num_procs,
					par->proc_name);
			for (k = 0; k < par->n; ++k) {
				printf("%3d ", par->c[k]);
			}
			printf("\n");
		}
		MPI_Barrier(MPI_COMM_WORLD );

	}
}

void PrintArrayC(params_t *par) {
	int k;
	printf("Proc %d/%d,%6s C ", par->myid, par->num_procs, par->proc_name);
	for (k = 0; k < par->n; ++k) {
		printf("%3d ", par->c[k]);
	}
	printf("\n");
}
void PrintArrayA(params_t *par) {
	int k;
	printf("Proc %d/%d,%6s A ", par->myid, par->num_procs, par->proc_name);
	for (k = 0; k < par->n; ++k) {
		printf("%3d ", par->a[k]);
	}
	printf("\n");
}

int myMPI_Scan(const void *sendbuf, void *recvbuf, int count,
		MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
	if (datatype != MPI_INT || op != MPI_SUM ) {
		fprintf(stderr, "Only MPI_INT with MPI_SUM are implemented");
		return -1;
	}
	memcpy(recvbuf, sendbuf, count * sizeof(int));
	MPI_Status status;
	int *tmp = (int*) calloc(count, sizeof(int));
	int *recbuf = (int*) recvbuf;
	int *senbuf = (int*) sendbuf;
	int levelstep2d = 1, levelstep2d1;
	int d, k, t;
	int m; // for inner loop for all elements
	int nodeleft, noderight;

	for (d = 1; d < par.nlevels; ++d) {
		levelstep2d1 = levelstep2d * 2;
		for (k = 0; k < par.nalloc; k += levelstep2d1) {
			nodeleft = k + levelstep2d - 1 - par.minidx;
			noderight = k + levelstep2d1 - 1 - par.minidx;
			if (nodeleft >= 0) {
				if (par.myid == nodeleft) {
					MPI_Send(recvbuf, count, MPI_INT, noderight, d, comm);
					break;
				}
				if (par.myid == noderight) {
					MPI_Recv(tmp, count, MPI_INT, nodeleft, d, comm, &status);
					for (m = 0; m < count; ++m)
						recbuf[m] += tmp[m];

					break;
				}
			}
		}
		levelstep2d *= 2;
	}
	MPI_Barrier(comm);
	if (par.myid == par.num_procs - 1) {
		for (m = 0; m < count; ++m)
			recbuf[m] = 0;
	}
	levelstep2d = par.nalloc / 2;

	for (d = par.nlevels - 1; d >= 0; --d) {
		levelstep2d1 = levelstep2d * 2;
		for (k = 0; k < par.nalloc; k += levelstep2d1) {
			nodeleft = k + levelstep2d - 1 - par.minidx;
			noderight = k + levelstep2d1 - 1 - par.minidx;
			if (nodeleft >= 0) {
				if (par.myid == nodeleft) {
					MPI_Send(recvbuf, count, MPI_INT, noderight, d, comm);
					MPI_Recv(recvbuf, count, MPI_INT, noderight, d, comm,
							&status);
				}
				if (par.myid == noderight) {
					MPI_Recv(tmp, count, MPI_INT, nodeleft, d, comm, &status);
					MPI_Send(recvbuf, count, MPI_INT, nodeleft, d, comm);
					for (m = 0; m < count; ++m)
						recbuf[m] += tmp[m];
				}

			}
		}
		levelstep2d /= 2;
	}
	for (m = 0; m < count; ++m)
		recbuf[m] += senbuf[m];
	free((void*) tmp);
	return 0;
}

void Init(int argc, char *argv[], params_t *par) {
	par->timer_global = par->timer_1 = par->timer_2 = par->timer_3 =
			par->timer_4 = 0;
	par->n = strtol(argv[1], NULL, 10);
	par->nlevels = (int) ceil(log(par->num_procs) / M_LN2);
	par->nalloc = (int) pow(2, par->nlevels);
	par->minidx = par->nalloc - par->num_procs;
	par->idx = par->myid + par->minidx;
	par->c = (int*) calloc((par->n), sizeof(int));
	par->a = (int*) calloc((par->n), sizeof(int));
	par->b = (int*) calloc((par->n), sizeof(int));

	int k;
	srand(MYRANDSEED + par->myid);
	for (k = 0; k < par->n; ++k) {
		par->c[k] = (rand() % RAND_RANGE) + 1;
	}
}

int SameOutput(int* a, int* b, int num, params_t *par) {

	int k, m;
	for (k = 0; k < par->num_procs; ++k) {
		MPI_Barrier(MPI_COMM_WORLD );
		if (par->myid != k)
			continue;
//		printf("Checking proc %d. \t", par->myid);
		for (m = 0; m < num; ++m) {
			if (a[m] != b[m])
				printf("a[%d]=%d \tb[%d]=%d \tc[%d]=%d \n", m, a[m], m, b[m], m,
						par->c[m]);
			break;
		}
//		printf("Same scan output at this node\n");
		MPI_Barrier(MPI_COMM_WORLD );
	}

	return -1;
}

void cleanup(params_t *par) {
	free((void*) par->a);
	free((void*) par->b);
	free((void*) par->c);
}
