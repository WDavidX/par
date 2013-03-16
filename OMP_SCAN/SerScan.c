#include "SerScan.h"
#include "util.c"
#include "omp.h"

#define VERBOSITY 1
#define NITERS          20

void cmd_parse(int argc, char *argv[], params_t *params);
void Scan_Serial_Seq(params_t * params);
void WriteOut(params_t * params);
void cleanup(params_t *params);
void OMP_Scan(params_t *params);

int main(int argc, char *argv[]) {
	params_t params;
	int k;
	printf("\nScan - Serial Version!!\n");
	cmd_parse(argc, argv, &params);
	gk_startwctimer(params.timer_1);
	for (k=0;k<1;++k)
		OMP_Scan(&params);
	gk_stopwctimer(params.timer_1);
//	WriteOut(&params);
	gk_stopwctimer(params.timer_global);
	printf("    wclock (sec): \t%.6lf\n", gk_getwctimer(params.timer_global));
	printf("    timer1 (sec): \t%.6lf\n", gk_getwctimer(params.timer_1));
	cleanup(&params);
	return 0;
}

void OMP_Scan(params_t *params){
	omp_set_num_threads(params->nthreads);
//		omp_set_num_threads(8);
	int nlevels=(int)ceil(log(params->nlines)/M_LN2);
	printf("nlevels=%d\t log(2)=%.6lf\n",nlevels,M_LN2);
	printf("OMP running on %d threads\n",omp_get_num_threads());
	int level;
#pragma omp parallel shared(nlevels,level)
	{
	int kk=1;
	printf("Inside %d, %d / %d\n",omp_get_thread_num(),omp_get_thread_num()+1,omp_get_num_threads());
	}
}


void cmd_parse(int argc, char *argv[], params_t *params) {
	gk_clearwctimer(params->timer_global);
	gk_clearwctimer(params->timer_1);
	gk_clearwctimer(params->timer_2);
	gk_clearwctimer(params->timer_3);
	gk_clearwctimer(params->timer_4);
	gk_startwctimer(params->timer_global);
	params->nthreads=strtol(argv[1], NULL, 10);
	params->infile = argv[2];
	size_t nlines, i;
	int x;
	FILE * fin;
	printf("%d arguments in the command line\n", argc);

	if ((fin = fopen(params->infile, "r")) == NULL ) {
		fprintf(stderr,"ERROR: Failed to open '%s' for reading\n",params->infile);
		exit(1);
	}
	if (fscanf(fin, "%zu\n", &nlines) != 1) {
		fprintf(stderr,"ERROR: Failed to read first line of '%s'\n",params->infile);
		fclose(fin);
		exit(2);
	}
	params->nlines = nlines;
	params->a = (int*)malloc(sizeof(int) * nlines);
	params->b = (int*)malloc(sizeof(int) * nlines);
	if (!params->a){
		fprintf(stderr,"Fail to allocate memory, a is a null %p\n",params->a);
	}
	if (!params->b){
			fprintf(stderr,"Fail to allocate memory, b is a null %p\n",params->a);
		}
	for (i = 0; i < nlines; ++i) {
		if (fscanf(fin, "%d\n", &x) != 1) {
			fprintf(stderr,"ERROR: Failed to read integer from line %zu/%zu in '%s'\n",
			i,nlines,params->infile);
			fclose(fin);
			exit(3);
		}
		params->a[i]=x;
	}
	fclose(fin);

	if (argc == 3) {
		params->outfile =(char*) malloc(sizeof(char) * 256);
		strcpy(params->outfile, "output.txt\0");
	} else {
		params->outfile = argv[4];
	}

	printf("File-in: %s \tfile-out: %s \tnlines=%zu \tnth=%d\n", params->infile,
			params->outfile, params->nlines,params->nthreads);
	memcpy(params->b,params->a,params->nlines*(sizeof(int)));
//	for (i=params->nlines-10;i<params->nlines;++i){
//		printf("a[%d]=%d \t b[%d]=%d\n",i,params->a[i],i,params->b[i]);
//	}
}

void Scan_Serial_Seq(params_t * params) {
	int i;
	params->b[i]=0;
	for (i = 1; i < params->nlines; ++i) {
		params->b[i] = params->b[i - 1] + params->a[i];
	}
}

void WriteOut(params_t * params) {
	FILE * fout;
	if ((fout = fopen(params->outfile, "w")) == NULL ) {
		fprintf(stderr,"ERROR: Failed to open '%s' for writing\n",params->outfile);
		exit(1);
	}
	int i;
	for (i = 0; i < params->nlines; ++i) {
		fprintf(fout, "%d\n", params->b[i]);
	}
	fclose(fout);
}

void cleanup(params_t *params){
	free((void*)params->a);
	free((void*)params->b);
}
