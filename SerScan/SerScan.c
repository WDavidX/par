#include "SerScan.h"
#include "util.c"


#define VERBOSITY 1

void cmd_parse(int argc, char *argv[], params_t *params);
void Scan_Serial_Seq(params_t * params);
void WriteOut(params_t * params);
void cleanup(params_t *params);


int main(int argc, char *argv[]) {
	params_t params;
	int k;
	printf("\nScan - Serial Version!!\n");
	cmd_parse(argc, argv, &params);
	gk_startwctimer(params.timer_1);
	for (k=0;k<10000;++k)
		Scan_Serial_Seq(&params);
	gk_stopwctimer(params.timer_1);
//	WriteOut(&params);
	gk_stopwctimer(params.timer_global);
	printf("    wclock (sec): \t%.2lf\n", gk_getwctimer(params.timer_global));
	printf("    timer1 (sec): \t%.2lf\n", gk_getwctimer(params.timer_1));
	cleanup(&params);
	return 0;
}

void cmd_parse(int argc, char *argv[], params_t *params) {
	gk_clearwctimer(params->timer_global);
	gk_clearwctimer(params->timer_1);
	gk_clearwctimer(params->timer_2);
	gk_clearwctimer(params->timer_3);
	gk_clearwctimer(params->timer_4);
	gk_startwctimer(params->timer_global);
	params->infile = argv[1];
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
	if (!params->a){
		fprintf(stderr,"Fail to allocate memory, a is a null %p\n",params->a);
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

	if (argc == 2) {
		params->outfile =(char*) malloc(sizeof(char) * 256);
		strcpy(params->outfile, "output.txt\0");
	} else {
		params->outfile = argv[3];
	}

	printf("File-in %s \tfile-out %s \tnlines=%zu\n", params->infile,
			params->outfile, params->nlines);
	params->b = (int*)malloc(sizeof(int) * (params->nlines));
	params->b[0] = params->a[0];
}

void Scan_Serial_Seq(params_t * params) {
	int i;
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
