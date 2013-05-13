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
#define RADBITS 3
#define RADIX (1<<RADBITS)

#define PASSHERE() do{ fprintf(stdout,"Passline %d >--------------------------< Passline %d\n",__LINE__,__LINE__);}while(0)
#define PF(x) do{ fprintf(stderr,"Line %d, %s=%.4f\n",__LINE__,#x,x);}while(0)
#define PINT(x) do{ fprintf(stderr,"Line %d, %s=%d\n",__LINE__,#x,x);}while(0)
#define PTR(x) do{ fprintf(stderr,"Line %d, %s=%p\n",__LINE__,#x,(void*)x);}while(0)
#define DPINT(x) do{ int localvaluetoshow;	cudaMemcpy((void*) &localvaluetoshow,(void*)&(x),sizeof(int),cudaMemcpyDeviceToHost);	fprintf(stderr,"Line %d, %s=%d\n",__LINE__,#x,localvaluetoshow);}while(0)
#define PSIMT(x) do{fprintf(stdout,"%d(%.1f) ",(x).pid,(x).sim.f);}while(0)
#define PSIMVAL(x) do{fprintf(stdout,"%.1f   ",(x).pid,(x).sim.f);}while(0)

#define wcint(a,n)\
	do{ int local_k;\
	fprintf(stdout,"Line %d, ArrayInt: %s   Number:%d \n",__LINE__, #a,n);\
		for (local_k=0;local_k<n;++local_k)\
			fprintf(stdout,"[%d]= %d   ",(local_k),(a[local_k]));\
			fprintf(stdout,"\n");\
	}while(0)

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
	int qid = blockIdx.x;
	int tid = threadIdx.x;
	int drow = dID + threadIdx.x;
	__shared__ sim_t
	out[CUDABLKNTH];
	/* Init output array */
	out[tid].pid = drow;
	out[tid].sim.f = 0.0;

	int qp, dp;
	int qrow = qID + blockIdx.x;

	int qnum = drowptr[qrow + 1] - drowptr[qrow];
	int dnum = drowptr[drow + 1] - drowptr[drow];
	__shared__
	int qind[CACHENUM];
	__shared__
	float qval[CACHENUM];
	if (tid < ndrows) {
		for (qp = tid; qp < qnum; qp += CUDABLKNTH) {
			qind[qp] = dcolind[drowptr[qrow] + qp];
			qval[qp] = dcolval[drowptr[qrow] + qp];
		}
	}
	__syncthreads();
	if (tid < ndrows) {
		qp = 0;
		dp = 0;
		while (qp < qnum && dp < dnum) {
			if (qind[qp] == dcolind[drowptr[drow] + dp]) {
				out[tid].sim.f += qval[qp] * dcolval[drowptr[drow] + dp] ;
				dp += 1;
				qp += 1;
			} else {
				if (qind[qp] < dcolind[drowptr[drow] + dp]) {
					qp += 1;
				} else {
					dp += 1;
				}
			}
		}
//		for (qp = 0, dp = 0; qp < qnum && dp < dnum; qp += 1) {
//			if (qind[qp] == dcolind[drowptr[drow] + dp]) {
//				out[tid].sim.f += qval[qp] * dcolval[drowptr[drow] + dp];
//				dp += 1;
//				while (qind[qp + 1] > dcolind[drowptr[drow] + dp] && dp < dnum)
//					dp += 1;
//			}
//		}
	}
	sim_t *dvout = &dout[qid * blockDim.x];  // get the desired row
	dvout[tid].pid = out[tid].pid;
	dvout[tid].sim.f = out[tid].sim.f;
	return;

}

void showa(sim_t *aa,int num){
		for (int k = 0; k < num; ++k) {
			PSIMVAL(aa[k]);
		}
		printf("\n");
	}

void radsort(sim_t *a, sim_t *b, int *c, int *counttmp, int num) {
	int pos, k, n;	
//	showa(a,num);	
//	printf("Radbits=%d Radix=%d\n",RADBITS,RADIX);
	for (pos = 0; pos < 32; pos += RADBITS) {
//		printf("\npos=%d\n",pos);
		memset((void*) c, 0, sizeof(int) * RADIX);
		for (k = 0; k < num; ++k) {
			c[(a[k].sim.i >> pos) % RADIX]++;
		}		
//		wcint(c,RADIX);
		for (k=1;k<RADIX;++k){
			c[k]=c[k]+c[k-1];
		}
//		wcint(c,RADIX);		
		for (k=num-1;k>=0;--k){
			n=(a[k].sim.i >> pos) % RADIX;
			c[n]--;	
			b[c[n]]=a[k];
//			PSIMT(a[k]);					
		}
//		printf("\n");
//		wcint(c,RADIX);
		memcpy(a, b, sizeof(sim_t) * num);
//		showa(a,num);
	}
}



#define NN (10)
void testrad() {
	sim_t a[NN], tmpa[NN];
	int count[RADIX];
	int counttmp[RADIX];
	int k;
	printf("\nStarted testing.\n");
	for (k = 0; k < NN; ++k) {
		a[k].pid = k;
//		a[k].sim.f = NN + 0.1 - k;
		a[k].sim.f = (NN + 4 - k) % NN + 0.1;
//		a[k].sim.f=k+0.1;
		PSIMT(a[k]);
	}
	printf("\n");
	radsort(a, tmpa, count, counttmp, NN);
	printf("\nAfter testing.\n");
	for (k = 0; k < NN; ++k) {
//		PSIMVAL(a[k]);
		PSIMT(a[k]);
	}
	printf("\n");

}

/*************************************************************************/
/*! Top-level routine for computing the neighbors of each document */
/**************************************************************************/
void ompComputeNeighbors(params_t *params) {
	testrad();
	exit(1);
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
//			printf("       Working on doc block %d/%d ndrows %d\n", dID, ndocs,
//					ndrows);
			Sim<<<nqrows,params->ndrows>>>
			(dcolind,dcolval,drowptr,qID,dID,params->nqrows,ndrows,dout);
			gk_stopwctimer(params->timer_3);

			for (qblkidx = 0; qblkidx < nqrows; ++qblkidx) {
				gk_startwctimer(params->timer_4);
				cudaMemcpy(hout[qblkidx], &dout[qblkidx * params->ndrows],
						sizeof(sim_t) * (params->ndrows), CPDH);
				gk_stopwctimer(params->timer_4);
				gk_startwctimer(params->timer_2);
				radsort(hout[qblkidx], tmpa, count, counttmp, houtnum);
				gk_stopwctimer(params->timer_2);
			}
		}

//		for (qblkidx = 0; qblkidx < nqrows; ++qblkidx) {
//			for (i = 0; i < params->nnbrs; ++i) {
//				dblkidx = houtnum -1- i;}}
		if (fpout) {
			for (qblkidx = 0; qblkidx < nqrows; ++qblkidx) {
				for (i = 0; i < params->nnbrs; ++i) {
					dblkidx = houtnum - 1 - i;
					if (hout[qblkidx][dblkidx].sim.f >= params->minsim) {
//						fprintf(stdout, "%3d $ %8d %8d %.3f  \n", i, qID + qblkidx,
//								hout[qblkidx][dblkidx].pid, hout[qblkidx][dblkidx].sim.f);
						fprintf(fpout, "%8d %8d %.3f  \n", qID + qblkidx,
								hout[qblkidx][dblkidx].pid, hout[qblkidx][dblkidx].sim.f);

					} else {
//					fprintf(stdout,"--->%3d $ %8d %8d %.3f  \n",i, qID + qblkidx, hout[qblkidx][dblkidx].pid,
//												hout[qblkidx][dblkidx].sim.f);
						break;
					}

				}
			}
		}

	}

	gk_stopwctimer(params->timer_1);

	/* cleanup and exit */
	if (fpout)
		gk_fclose(fpout);

	PASSHERE()
	;
	printf("#docs: %d, #nnz: %d.\n", vault->ndocs,
			vault->mat->rowptr[vault->mat->nrows]);

	iserr()
	;
	FreeVault(vault);
	return;
}

__device__ void testAdd(int *a, int *b, int *c) {
	*c = *a + *b;
}

