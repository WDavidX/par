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
//#define CUDABLKNTH (4)
#define NUMQROWS (2048)
#define CACHENUM (2048)
//#define NUMQROWS (1)
#define RADBITS 8
#define RADIX (1<<RADBITS)

#define PASSHERE() do{ fprintf(stdout,"Passline %d >--------------------------< Passline %d\n",__LINE__,__LINE__);}while(0)
#define PF(x) do{ fprintf(stderr,"Line %d, %s=%.4f\n",__LINE__,#x,x);}while(0)
#define PINT(x) do{ fprintf(stderr,"Line %d, %s=%d\n",__LINE__,#x,x);}while(0)
#define PTR(x) do{ fprintf(stderr,"Line %d, %s=%p\n",__LINE__,#x,(void*)x);}while(0)
#define DPINT(x) do{ int localvaluetoshow;	cudaMemcpy((void*) &localvaluetoshow,(void*)&(x),sizeof(int),cudaMemcpyDeviceToHost);	fprintf(stderr,"Line %d, %s=%d\n",__LINE__,#x,localvaluetoshow);}while(0)
#define PSIMT(x) do{fprintf(stdout,"%d(%.1f) ",(x).pid,(x).sim.f);}while(0)

#define iserr() do{ cudaError_t localcudaerror=cudaGetLastError(); if (cudaSuccess!=localcudaerror) {printf("!!! CUDA err Line %d: %d %s\n",__LINE__,localcudaerror,cudaGetErrorString(localcudaerror));} else {printf("CUDA OK Line %d\n",__LINE__);}}while(0)

#define peekerr() do{ cudaError_t localcudaerror=cudaPeekAtLastError(); if (cudaSuccess!=localcudaerror) {printf("!!! CUDA err Line %d: %d %s\n",__LINE__,localcudaerror,cudaGetErrorString(localcudaerror));} else {printf("CUDA OK Line %d\n",__LINE__);}}while(0)

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
#define wwait(x) do{usleep(x*1000000);} while(0)

__global__ void Sim(int *dcolind, float *dcolval, int *drowptr, int qID,
		int dID, int parnqrows, int ndrows, sim_t *dout) {
	int qrow = qID + blockIdx.x;
	int drow = dID + threadIdx.x;
	int qid = blockIdx.x;
	int tid = threadIdx.x;
	int qnum = drowptr[qrow + 1] - drowptr[qrow];
	int dnum = drowptr[drow + 1] - drowptr[drow];
	__shared__
	int qind[CACHENUM];
	__shared__
	float qval[CACHENUM];
	__shared__ sim_t
	out[CUDABLKNTH];
	/* Init output array */
	out[tid].pid = drow;
	out[tid].sim.f = 0.0;
	if (tid < ndrows) {
		int qp, dp;
		for (qp = tid; qp < qnum; qp += CUDABLKNTH) {
			qind[qp] = dcolind[drowptr[qrow] + qp];
			qval[qp] = dcolval[drowptr[qrow] + qp];
		}
		__syncthreads();
		for (qp = 0, dp = 0; qp < qnum && dp < dnum; qp += 1) {
			if (qind[qp] == dcolind[drowptr[drow] + dp]) {
				out[tid].sim.f += qval[qp] * dcolval[drowptr[drow] + dp];
				dp += 1;
				while (qind[qp + 1] > dcolind[drowptr[drow] + dp] && dp < dnum)
					dp += 1;
			}
		}
	}
	sim_t *dvout = &dout[qid * blockDim.x];  // get the desired row
	dvout[tid].pid = out[tid].pid;
	dvout[tid].sim.f = out[tid].sim.f;
	return;

}

void radsort(sim_t *a, sim_t *tmpa, int *count, int *counttmp, int num) {
	int pos, k, n;
//
//	printf("\nInit State\n");
//	for (k = 0; k < num; ++k) {
//		if (a[k].sim.f > 0.1) {
//			printf("%d$", k);
//			PSIMT(a[k]);
//		}
//	}
//	printf("\n");

	for (pos = 0; pos < 32; pos += RADBITS) {
		memset((void*) count, 0, sizeof(int) * RADIX);
		for (k = 0; k < num; ++k) {
			count[((a[k].sim.i) >> pos) % RADIX]++;
		}
		counttmp[0] = 0;
		for (k = 1; k < RADIX; ++k) {
			counttmp[k] = counttmp[k - 1] + count[k - 1];
		}
		for (k = 0; k < num; ++k) {
			n = ((a[k].sim.i) >> pos) % RADIX;
			tmpa[counttmp[n]] = a[k];
			counttmp[n]++;
		}

//		printf("\npos=%d pass=%d/%d\n", pos, pos / RADBITS + 1, 32 / RADBITS);
//		for (k = 0; k < num; ++k) {
//			if (tmpa[k].sim.f > 0.1) {
//				printf("%d$", k);
//				PSIMT(tmpa[k]);
//			}
//		}
//		printf("\n");

		memcpy(a, tmpa, sizeof(sim_t) * num);
	}
}

#define NN (1<<10)
void testrad() {
	sim_t a[NN], tmpa[NN];
	int count[RADIX];
	int counttmp[RADIX];
	int k;
	for (k = 0; k < NN; ++k) {
		a[k].pid = k;
		a[k].sim.f = NN + 0.1 - k;
		PSIMT(a[k]);
	}
	printf("\nStarted testing.\n");
	for (k = 0; k < NN; ++k) {
		a[k].pid = k;
		a[k].sim.f = NN + 0.1 - k;
		PSIMT(a[k]);
	}
	printf("\n");
	radsort(a, tmpa, count, counttmp, NN);

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
	gk_csr_t *mat = vault->mat;

	params->endid = (params->endid == -1 ? vault->mat->nrows : params->endid);
	printf("#docs: %d, #nnz: %d.\n", vault->ndocs,
			vault->mat->rowptr[vault->mat->nrows]);

	/* My stuff starts from here */
	/* reset the nqrows and ndrows */

	params->nqrows = NUMQROWS;
	params->ndrows = CUDABLKNTH;
	int ndocs = vault->ndocs, nnz = vault->mat->rowptr[vault->mat->nrows];
	printf("nqrows: %d, ndrows: %d.\n", params->nqrows, params->ndrows);
	omp_set_num_threads(8);
	gk_csr_CompactColumns(vault->mat);
	gk_csr_Normalize(vault->mat, GK_CSR_ROW, 2);
	/* create the output file */
	fpout = (
			params->outfile ?
					gk_fopen(params->outfile, "w", "ComputeNeighbors: fpout") : NULL);

	gk_startwctimer(params->timer_1);
	/* my stuff */

	int *drowptr, *dcolind;
	float *dcolval;
	cudaFree((void**) NULL);
	gk_startwctimer(params->timer_3);

	cudaMalloc((void**) &dcolind, sizeof(int) * (nnz));
	cudaMemcpy(dcolind, mat->rowind, sizeof(int) * (nnz), CPHD);

	cudaMalloc((void**) &drowptr, sizeof(int) * (vault->ndocs + 1));
	cudaMemcpy((void*) drowptr, (void*) mat->rowptr, sizeof(int) * (ndocs + 1),
			CPHD);

	cudaMalloc((void**) &dcolval, sizeof(float) * (nnz));
	cudaMemcpy(dcolval, mat->rowval, sizeof(float) * (nnz), CPHD);

	gk_stopwctimer(params->timer_3);
	printf("    timer3:  CPHD   (sec):   %.2lf\n", params->timer_3);

	/* ************ copy mat to device ****** */
	sim_t *dout;
	sim_t *houtsave;
	int qblkidx, dblkidx, *tmpint, houtnum = params->ndrows + params->nnbrs;

	cudaMalloc((void**) &dout, sizeof(sim_t) * (params->nqrows * params->ndrows));
	wccalloc(houtsave, sim_t,
			sizeof(sim_t) * (params->nqrows * (params->ndrows + params->nnbrs)));
	sim_t **hout;
	wcalloc(hout, sim_t*, sizeof(sim_t*) * (params->nqrows));
	for (qblkidx = 0; qblkidx < params->nqrows; qblkidx += 1)
		hout[qblkidx] = &houtsave[qblkidx * (params->ndrows + params->nnbrs)];

	/* ************ radix sort init ****** */
	int *count, *counttmp;
	sim_t *tmpa;
	wcalloc(count, int, (RADIX));
	wcalloc(counttmp, int, (RADIX));
	wcalloc(tmpa, sim_t, houtnum);
	for (qID = params->startid; qID < params->endid; qID += params->nqrows) {
		memset(houtsave, 0, sizeof(sim_t) * (params->nqrows * houtnum));
		nqrows = gk_min(params->nqrows, params->endid - qID);
		printf(" Working on query block %d/%d nqrows %d\n", qID, ndocs, nqrows);
		/* find the neighbors of the chunk */
		for (dID = 0; dID < vault->ndocs; dID += params->ndrows) {
			ndrows = gk_min(params->ndrows, vault->ndocs - dID);
			dblkidx = dID / params->ndrows;
			gk_startwctimer(params->timer_3);
			printf("       Working on doc block %d/%d ndrows %d\n", dID, ndocs,
					ndrows);
			Sim<<<nqrows,params->ndrows>>>
			(dcolind,dcolval,drowptr,qID,dID,params->nqrows,ndrows,dout);
			gk_stopwctimer(params->timer_3);
			for (qblkidx = 0; qblkidx < nqrows; ++qblkidx) {
				gk_startwctimer(params->timer_4);
				cudaMemcpy(hout[qblkidx], &dout[qblkidx * params->ndrows],
						sizeof(sim_t) * (params->ndrows), CPDH);
				gk_stopwctimer(params->timer_4);
//				if (qblkidx == 0 && qID == 0 || 1) {
//					printf(
//							"\n---------- qID=%d dID=%d qblkidx =%d ndrows=%d ----------\n",
//							qID, dID, qblkidx, ndrows);
//					for (dblkidx = 0; dblkidx < ndrows; ++dblkidx) {
//						if (hout[qblkidx][dblkidx].sim.i != 0)
//							printf("%d->%d (%.3f)  ", qID + qblkidx,
//									hout[qblkidx][dblkidx].pid, hout[qblkidx][dblkidx].sim.f);
//					}
//
//				}
//
//				printf("\nUnsorted ---> queryid=%d\n", qID + qblkidx);
//
//				for (dblkidx = 0; dblkidx < ndrows; ++dblkidx) {
//					if (hout[qblkidx][dblkidx].sim.f > params->minsim)
//						printf("%d->%d (%.3f)  ", qID + qblkidx, hout[qblkidx][dblkidx].pid,
//								hout[qblkidx][dblkidx].sim.f);
//				}

				gk_startwctimer(params->timer_2);
				radsort(hout[qblkidx], tmpa, count, counttmp, houtnum);
				gk_stopwctimer(params->timer_2);

				printf("Sorted ---> queryid=%d\n", qID + qblkidx);
				for (i = 0; i < params->nnbrs; ++i) {
					dblkidx = houtnum - i;
					if (hout[qblkidx][dblkidx].sim.f > params->minsim)
						printf("%d->%d (%.3f)  \n", qID + qblkidx,
								hout[qblkidx][dblkidx].pid, hout[qblkidx][dblkidx].sim.f);
				}

//				printf("Sorted ---> queryid=%d\n", qID + qblkidx);
//				for (dblkidx = 0; dblkidx < ndrows; ++dblkidx) {}
//				for (dblkidx = 0; dblkidx < ndrows; ++dblkidx) {
//					if (hout[qblkidx][dblkidx].sim.f > params->minsim)
//						printf("%d->%d (%.3f)  ", qID + qblkidx, hout[qblkidx][dblkidx].pid,
//								hout[qblkidx][dblkidx].sim.f);
//				}
//				for (i = 0; i < params->nnbrs; ++i) {
//					dblkidx = houtnum - i;
//					if (hout[qblkidx][dblkidx].sim.f > params->minsim)
//						printf("%d->%d (%.3f)  ", qID + qblkidx, hout[qblkidx][dblkidx].pid,
//								hout[qblkidx][dblkidx].sim.f);
//				}
//				for (dblkidx = houtnum-1; dblkidx > houtnum-params->nnbrs; --dblkidx) {
//					if (hout[qblkidx][dblkidx].sim.f >params->minsim)
//						printf("%d->%d (%.3f)  ", qID + qblkidx,
//								hout[qblkidx][dblkidx].pid, hout[qblkidx][dblkidx].sim.f);
//				}
//				printf("\n", qID + qblkidx);
			}
		}
		printf("fpout = %p\n", fpout);
		if (fpout) {
			for (i = 0; i < nqrows; i++) {
				for (j = houtnum - 1;
						j < houtnum - params->nnbrs && hout[i][j].sim.f >= params->minsim;
						j--) {
					fprintf(stdout, "%8d %8d %.3f\n", qID + i, hout[i][j].pid,
							hout[i][j].sim.f);
				}
			}
		}
	}

	gk_stopwctimer(params->timer_1);

	/* cleanup and exit */
	if (fpout)
		gk_fclose(fpout);

	FreeVault(vault);
	PASSHERE()
	;
	iserr()
	;
	return;
}

__device__ void testAdd(int *a, int *b, int *c) {
	*c = *a + *b;
}
