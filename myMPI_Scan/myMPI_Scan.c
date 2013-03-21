#include <mpi.h>
#include "myMPI_Scan.h"

#define Tclear(tmr) (tmr = 0.0)
#define Tstart(tmr) (tmr -= MPI_Wtime())
#define Tstop(tmr)  (tmr += MPI_Wtime())
#define Tget(tmr)   (tmr)

//#define RAND_RANGE INT_MAX
#define RAND_RANGE 10

void Init(int argc, char *argv[], params_t *par);
void myMPI_Scan();
void cleanup(params_t *par);

int SameOutput(int* a, int* b, size_t num);

void PrintArray(params_t *par);

int main(int argc, char *argv[]) {
	params_t par;
	MPI_Init(&argc, &argv);
	int num_procs, myid, name_len;
	char proc_name[MPI_MAX_PROCESSOR_NAME];
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Get_processor_name(proc_name, &name_len);
	printf("Proc %d/%d,%6s\n", myid, num_procs, proc_name);
	par.num_procs = num_procs;
	par.myid = myid;
	par.name_len = name_len;
	par.proc_name = (char*) &proc_name;
	Init(argc, argv, &par);

	MPI_Barrier(MPI_COMM_WORLD );
	Tstart(par.timer_1);
	MPI_Scan(par.a, par.a, par.n, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
	Tstop(par.timer_1);
	MPI_Barrier(MPI_COMM_WORLD );

	PrintArray(&par);
	MPI_Barrier(MPI_COMM_WORLD );
	// Last call to MPI (REQUIRED)
	MPI_Finalize();
	return 0;
}

void PrintArray(params_t *par) {
//	MPI_Win win;
//	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, par->myid, 0, win);
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
//	MPI_Win_unlock(par->myid,win);
}

void myMPI_Scan() {

}

void Init(int argc, char *argv[], params_t *par) {
	par->timer_global = par->timer_1 = par->timer_2 = par->timer_3 =
			par->timer_4 = 0;
	par->n = strtol(argv[1], NULL, 10);
	par->nlevels = (int) ceil(log(par->n) / M_LN2);
	par->nalloc = (int) (pow(2, par->nlevels));

	par->c = (int*) calloc((1 + par->nalloc), sizeof(int));
	par->a = (int*) calloc((1 + par->nalloc), sizeof(int));
	par->b = (int*) calloc((1 + par->nalloc), sizeof(int));
	int k;
	srand(time(NULL ));
	for (k = 0; k < par->n; ++k) {
		par->c[k] = (rand() % RAND_RANGE)+1;
		par->b[k] = par->c[k];
		par->a[k] = par->c[k];
	}

}

int SameOutput(int* a, int* b, size_t num) {
	return 1;
}

void cleanup(params_t *par) {
	free((void*) par->a);
	free((void*) par->b);
	free((void*) par->c);
}
