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

int SameOutput(int* a, int* b, int num,params_t *par);

void PrintArray(params_t *par);
void PrintArrayC(params_t *par);
void PrintArrayA(params_t *par);

void InsertNewLine(params_t *par){
	MPI_Barrier(MPI_COMM_WORLD );
	if (par->myid==0){
		printf("\n");
	}
	MPI_Barrier(MPI_COMM_WORLD );
}

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
	MPI_Scan(par.c, par.a, par.n, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
	Tstop(par.timer_1);
	MPI_Barrier(MPI_COMM_WORLD );

	MPI_Barrier(MPI_COMM_WORLD );
	Tstart(par.timer_2);
//	MPI_Scan(par.c, par.b, par.n, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
	Tstop(par.timer_2);
	MPI_Barrier(MPI_COMM_WORLD );

	PrintArray(&par);
	// Last call to MPI (REQUIRED)

	SameOutput(par.a,par.b,par.n,&par);

	MPI_Barrier(MPI_COMM_WORLD );
	if (par.myid==par.num_procs-1){

		printf("\n  OMP_SCAN timer1 (sec): \t %.7lf",par.timer_1);
		printf("\nmyOMP_SCAN timer2 (sec): \t %.7lf",par.timer_2);
	}
	MPI_Barrier(MPI_COMM_WORLD );

	MPI_Finalize();
	return 0;
}

void PrintArray(params_t *par) {
//	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, par->myid, 0, win);

	int prc_num;

	for (prc_num=0;prc_num<par->num_procs;++prc_num){
		MPI_Barrier(MPI_COMM_WORLD );
		if (par->myid==prc_num){
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


//	MPI_Win_unlock(par->myid,win);
}

void PrintArrayC(params_t *par) {
	int k;
	printf("Proc %d/%d,%6s C ", par->myid, par->num_procs,
			par->proc_name);
	for (k = 0; k < par->n; ++k) {
		printf("%3d ", par->c[k]);
	}
	printf("\n");
}
void PrintArrayA(params_t *par) {
	int k;
	printf("Proc %d/%d,%6s A ", par->myid, par->num_procs,
			par->proc_name);
	for (k = 0; k < par->n; ++k) {
		printf("%3d ", par->a[k]);
	}
	printf("\n");
}

void myMPI_Scan() {

}

void Init(int argc, char *argv[], params_t *par) {
	par->timer_global = par->timer_1 = par->timer_2 = par->timer_3 =
			par->timer_4 = 0;
	par->n = strtol(argv[1], NULL, 10);
	par->nlevels = (int) ceil(log(par->num_procs) / M_LN2);
	par->nalloc = par->n ;

	par->c = (int*) calloc(( par->nalloc), sizeof(int));
	par->a = (int*) calloc(( par->nalloc), sizeof(int));
	par->b = (int*) calloc(( par->nalloc), sizeof(int));
	int k;
	srand(time(NULL)+par->myid);
	for (k = 0; k < par->n; ++k) {
		par->c[k] = (rand() % RAND_RANGE)+1;
	}

}

int SameOutput(int* a, int* b, int num,params_t *par){
	if (par->myid!=par->num_procs-1)
		return -2;
	if (par->myid==par->num_procs-1){
		int k;
		for (k=0;k<num;++k){
			if (a[k]!=b[k])
				printf("a[%d]=%d\nb[%d]=%d\n",k,a[k],k,b[k]);
				return k;
		}
	}
	printf("Same output for A and B\n");
	return -1;
}

void cleanup(params_t *par) {
	free((void*) par->a);
	free((void*) par->b);
	free((void*) par->c);
}
