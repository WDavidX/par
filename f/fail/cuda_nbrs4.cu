/*!
 \file  ompnbrs.c
 \brief Computes the nearest neighbors of each document 
 
 \date   Started 2/3/10
 \author George
 \version\verbatim $Id: omp_nbrs.c 9585 2011-03-18 16:51:51Z karypis $ \endverbatim
 */

extern "C" {
#include "simdocs.h"
}
#define NITERS          20

#define CPDH cudaMemcpyDeviceToHost
#define CPHD cudaMemcpyHostToDevice

#define CUDABLKNTH (1024)
#define NUMQROWS (2048)
#define CACHENUM (2048)
//#define NUMQROWS (1)

#define PASSHERE() do{ fprintf(stdout,"Passline %d >--------------------------< Passline %d\n",__LINE__,__LINE__);}while(0)
#define PINT(x) do{ fprintf(stderr,"Line %d, %s=%d\n",__LINE__,#x,x);}while(0)
#define DPINT(x) do{ int localvaluetoshow;\
		cudaMemcpy((void*) &localvaluetoshow,(void*)&(x),sizeof(int),cudaMemcpyDeviceToHost);\
		fprintf(stderr,"Line %d, %s=%d\n",__LINE__,#x,localvaluetoshow);}while(0)

#define wcalloc(PTR,TYPE, ALLOCNUM)\
	do{\
	PTR=(TYPE*)malloc(sizeof(TYPE)*(ALLOCNUM));\
	if ((PTR)==NULL)\
		fprintf(stderr,"Malloc failure PTR=%s @ Line %d\t\n",#PTR,__LINE__);\
	}while(0)
#define wccalloc(PTR,TYPE, ALLOCNUM)\
	do{\
	PTR=(TYPE*)calloc(sizeof(TYPE),(ALLOCNUM));\
	if ((PTR)==NULL)\
		fprintf(stderr,"Malloc failure PTR=%s @ Line %d\t\n",#PTR,__LINE__);\
	}while(0)
#define CPHTD(dst,src,num) do{ cudaMemcpy((void*)(dst),(void*) (src), (num),cudaMemcpyHostToDevice);} while(0)
#define CPDTH(dst,src,num) do{ cudaMemcpy((void*)(dst),(void*) (src), (num),cudaMemcpyDeviceToHost);} while(0)
#define wwait(x) do{long int locali,localk; for (locali=0;locali<(x)*1000000/4;++locali){localk=locali;};} while(0)

__global__ void cudaCalcSim(int *dcolind, float *dcolval, int *drowptr, int qID,
		int dID, int nqrows, int ndrows, int querycache, sim_t **dout) {
	int cachelen = querycache;

	int qrow = qID + blockIdx.x;
	int drow = dID + threadIdx.x;

	int qid = blockIdx.x;
	int tid = threadIdx.x;
	int qnum = drowptr[qrow + 1] - drowptr[qrow];
	int dnum = drowptr[drow + 1] - drowptr[drow];

	__shared__
	int qind[CACHENUM];
	__shared__
	int dind[CACHENUM];
	__shared__
	float qval[CACHENUM];
	__shared__
	float dval[CACHENUM];
	__shared__ sim_t
	out[CUDABLKNTH];
	/* Init output array */
	out[tid].pid = drow;
	out[tid].sim.f = 0.0;
	/* calc output */
	int qp, dp;
	for (qp = tid; qp < qnum; qp += CUDABLKNTH) {
		qind[qp] = dcolind[drowptr[qrow] + qp];
		qval[qp] = dcolind[drowptr[qrow] + qp];
	}
	__syncthreads();
	qp = 0;
	dp = 0;
//	while (qp<qnum && dp<dnum){
//		if (qind[qp]==dcolind[drowptr[qrow]+dp])
//	}
//	for (qp = 0, dp = 0; qp < qnum && dp < dnum; qp += 1) {
//		if (qind[qp] == dcolind[drowptr[qrow] + qp]) {
//			out[tid].sim.f += qval[qp] * dcolval[drowptr[qrow] + qp];
//			dp += 1;
//			while (qind[qp + 1] > dcolind[drowptr[drow] + dp]&& dp < dnum)
//				dp += 1;
//		}
//	}

//	dout[qid][tid].pid = out[tid].pid;
//	dout[qid][tid].sim.f = out[tid].sim.f;
//	printf("\n------------------------------\n");
//	printf("\nqid=%d, tid=%d, querynum=%d\n",qid,tid,querynum);
	return;

}

/*************************************************************************/
/*! Top-level routine for computing the neighbors of each document */
/**************************************************************************/
void ompComputeNeighbors(params_t *params) {
	int i, j, qID, dID, nqrows, ndrows;
	vault_t *vault;
	//gk_csr_t *mat;
	FILE *fpout;
	printf("Reading data for %s...\n", params->infstem);
	vault = ReadData(params);
	params->endid = (params->endid == -1 ? vault->mat->nrows : params->endid);
	printf("#docs: %d, #nnz: %d.\n", vault->ndocs,
			vault->mat->rowptr[vault->mat->nrows]);

	/* My stuff starts from here */
	/* reset the nqrows and ndrows */

	params->nqrows = NUMQROWS;
	params->ndrows = CUDABLKNTH;
	printf("nqrows: %d, ndrows: %d.\n", params->nqrows, params->ndrows);
	/* Compact the column-space of the matrices */
	gk_csr_CompactColumns(vault->mat);

	/* Perform auxiliary normalizations/pre-computations based on similarity */
	gk_csr_Normalize(vault->mat, GK_CSR_ROW, 2);
	/* Index pmat once for all */
	int ndocs = vault->ndocs, nnz = vault->mat->rowptr[vault->mat->nrows];
	int *drowptr;
	int *dcolind;
	float *dcolval;
	cudaFree((void**) NULL);
	gk_startwctimer(params->timer_3);
	cudaMalloc((void**) &drowptr, sizeof(int) * (vault->ndocs + 1));
	cudaMemcpy(drowptr, vault->mat->rowptr, sizeof(int) * (ndocs + 1), CPHD);

	cudaMalloc((void**) &dcolind, sizeof(int) * (nnz));
	cudaMemcpy(dcolind, vault->mat->colind, sizeof(int) * (nnz), CPHD);

	cudaMalloc((void**) &dcolval, sizeof(float) * (nnz));
	cudaMemcpy(dcolval, vault->mat->colval, sizeof(int) * (nnz), CPHD);
	gk_stopwctimer(params->timer_3);

	sim_t **hout, **dout;
	wcalloc(hout, sim_t*, params->nqrows);
	wcalloc(dout, sim_t*, params->nqrows);

	int qblkid, dblkid;
	int maxcolnum = -1, rowwithmaxcolnum = -1, querylength;
	for (qID = 0; qID < vault->ndocs; qID += 1) {
		querylength = vault->mat->rowptr[qID + 1] - vault->mat->rowptr[qID];
		if (maxcolnum < querylength) {
			maxcolnum = querylength;
			rowwithmaxcolnum = qID;
		}
	}
	printf("Longest row %d with %d entries\n", rowwithmaxcolnum, maxcolnum);
	gk_startwctimer(params->timer_2);
	wwait(1000);
	gk_stopwctimer(params->timer_2);

	for (qID = 0; qID < params->nqrows; qID += 1) {
		qblkid = qID;
		gk_startwctimer(params->timer_3);
		cudaMalloc((void**) &dout[qblkid], sizeof(sim_t) * (params->ndrows));
		gk_stopwctimer(params->timer_3);
		wcalloc(hout[qblkid], sim_t, params->nnbrs);
	}
	printf("    timer3:  CPHD   (sec):   %.2lf\n",
			gk_getwctimer(params->timer_3));
	/* create the output file */
	fpout = (
			params->outfile ?
					gk_fopen(params->outfile, "w", "ComputeNeighbors: fpout") :
					NULL);

	/* allocate memory for the working sets */

	gk_startwctimer(params->timer_1);

	omp_set_num_threads(8);

	/* break the computations into chunks */
	for (qID = params->startid; qID < params->endid; qID += params->nqrows) {
		nqrows = gk_min(params->nqrows, params->endid - qID);
		qblkid = qID / params->nqrows;
		for (dID = 0; dID < vault->ndocs; dID += params->ndrows) {
			ndrows = gk_min(params->ndrows, vault->ndocs - dID);
			dblkid = dID / params->ndrows;
			gk_startwctimer(params->timer_3);
//			cudaCalcSim<<<nqrows,ndrows>>>(dcolind,dcolval,drowptr,qID,dID,nqrows,ndrows,maxcolnum,dout);
			gk_stopwctimer(params->timer_3);

		}

	}

	gk_stopwctimer(params->timer_1);

	/* cleanup and exit */
	if (fpout)
		gk_fclose(fpout);
	FreeVault(vault);

	return;
}

__device__ void testAdd(int *a, int *b, int *c) {
	*c = *a + *b;
}
