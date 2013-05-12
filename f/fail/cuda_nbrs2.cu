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

#define CUDABLKNTH (1)
#define NUMQROWS (2048)
#define LOAD_QUERY (49152/4/NUMQROWS)
//#define NUMQROWS (1)

#define PASSHERE() do{ fprintf(stderr,"Passline %d\n",__LINE__);}while(0)
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

void cudaFindNeighbors(params_t *params, vault_t *vault, int qID, int nqrows,
		int dID, int ndrows, int *nallhits, sim_t **allhits);

int gk_csr_cuda_GetSimilarRows(gk_csr_t *mat, int nqterms, int *qind,
		float *qval, int simtype, int nsim, float minsim, gk_fkv_t *hits,
		int *i_marker, gk_fkv_t *i_cand);

__global__ void cudaSimilarity(int qstartid, gk_csr_t *mat, int *dcolptr,
		int *drowind, float *drowval, int docnr, int docnnz, int nnbr,
		int numshared, int *mynallhits, sim_t **myallhits);

/*************************************************************************/
/*! Top-level routine for computing the neighbors of each document */
/**************************************************************************/
void ompComputeNeighbors(params_t *params) {
	int i, j, qID, dID, nqrows, ndrows;
	vault_t *vault;
	sim_t **allhits;
	int *nallhits;
	FILE *fpout;

	params->nqrows = NUMQROWS;
//	params->ndrows = CUDABLKNTH - params->nnbrs;
	params->ndrows = CUDABLKNTH;
	printf("Reading data for %s...\n", params->infstem);
	vault = ReadData(params);
	params->endid = (params->endid == -1 ? vault->mat->nrows : params->endid);
	printf("#docs: %d, #nnz: %d.\n", vault->ndocs,
			vault->mat->rowptr[vault->mat->nrows]);
	printf("nqrows: %d, ndrows: %d. \nBLK shared int/float %d (%.3f)\n",
			params->nqrows, params->ndrows, LOAD_QUERY,
			(49152 / 4.0 / NUMQROWS));
	omp_set_num_threads(8);
	/* Compact the column-space of the matrices */
	gk_csr_CompactColumns(vault->mat);

	/* Perform auxiliary normalizations/pre-computations based on similarity */
	gk_csr_Normalize(vault->mat, GK_CSR_ROW, 2);

	/* create the output file */
	fpout = (
			params->outfile ?
					gk_fopen(params->outfile, "w", "ComputeNeighbors: fpout") :
					NULL);
	/* index all doc matrices */
	gk_csr_t **pmatidx;
	gk_csr_t *hpmat;
//	cudaMalloc((void**) *dout,sizeof(sim_t)*)
	sim_t **hout;
	wcalloc(hout, sim_t*, params->nqrows);
	for (qID = 0; qID < params->nqrows; ++qID) {
		wccalloc(hout[qID], sim_t, params->ndrows + params->nnbrs);
	}
	int **dcolptr, **drowind;
	float **drowval;
	int *ddcolptr, *ddrowind;
	float *ddrowval;
	int blkid;
	int docnr, docnnz;
	wcalloc(pmatidx, gk_csr_t*, (vault->ndocs / params->ndrows + 1)); //need to free
	wcalloc(dcolptr, int*, (vault->ndocs / params->ndrows + 1)); //need to free
	wcalloc(drowind, int*, (vault->ndocs / params->ndrows + 1)); //need to free
	wcalloc(drowval, float*, (vault->ndocs / params->ndrows + 1)); //need to free

	for (dID = 0; dID < vault->ndocs; dID += params->ndrows) {
		ndrows = gk_min(params->ndrows, vault->ndocs - dID);
		blkid = dID / params->ndrows;
		gk_startwctimer(params->timer_2);
		pmatidx[blkid] = gk_csr_ExtractSubmatrix(vault->mat, dID, ndrows);
		ASSERT(pmatidx[blkid] != NULL);
		gk_stopwctimer(params->timer_2);
		gk_startwctimer(params->timer_4);
		gk_csr_CreateIndex(pmatidx[blkid], GK_CSR_COL);
		gk_stopwctimer(params->timer_4);
		hpmat = pmatidx[blkid];
		gk_startwctimer(params->timer_3);
		docnr = hpmat->nrows;
		docnnz = hpmat->rowind[docnr];
		ddcolptr = dcolptr[blkid];
		ddrowind = drowind[blkid];
		ddrowval = drowval[blkid];
		cudaMalloc((void**) &ddcolptr, sizeof(int) * (docnr + 1));
		cudaMalloc((void**) &ddrowind, sizeof(int) * (docnnz));
		cudaMalloc((void**) &ddrowval, sizeof(float) * (docnnz));
		cudaMemcpy(ddcolptr, hpmat->colptr, sizeof(int) * (docnr + 1), CPHD);
		cudaMemcpy(ddrowind, hpmat->rowind, sizeof(int) * (docnnz), CPHD);
		cudaMemcpy(ddrowval, hpmat->rowval, sizeof(float) * (docnnz), CPHD);
		gk_stopwctimer(params->timer_3);

	}
	printf("CUDA malloc pmat (sec): %.2f \n", params->timer_3);
	printf("Indexed %d %d-row submatrices.(%d/%d)\n",
			vault->ndocs / params->ndrows + 1, params->ndrows,
			params->ndrows + vault->ndocs / params->ndrows * params->ndrows,
			vault->ndocs);
	/* allocate memory for the working sets */

	gk_AllocMatrix((void ***) &allhits, sizeof(sim_t), params->nqrows,
			2 * params->nnbrs);
	nallhits = gk_imalloc(params->nqrows, "gComputeNeighbors: nallhits");

	gk_startwctimer(params->timer_1);

	/* Varibles */
	cudaFree((void*) NULL);
	sim_t **myallhits;
	int *mynallhits;
	myallhits = (sim_t **) malloc(sizeof(sim_t*) * params->nqrows);
	mynallhits = (int*) calloc(sizeof(int), params->nqrows);
	for (i = 0; i < params->nqrows; ++i) {
		cudaMalloc((void**) &myallhits[i], sizeof(sim_t) * CUDABLKNTH);
	}
	// for pmat

	int maxcolnum = -1, n;
	for (qID = params->startid; qID < params->endid; qID += 1) {
		int n=vault
		->mat->rowptr[qID+1]-vault->mat->rowptr[qID];
		if (maxcolnum < n && n != 1172)
			maxcolnum = n;
	}
	printf("Max row none zero number %d\n", maxcolnum);

	/* break the computations into chunks */
	for (qID = params->startid; qID < params->endid; qID += params->nqrows) {
		nqrows = gk_min(params->nqrows, params->endid - qID);
		gk_iset(nqrows, 0, nallhits);
//		printf("Working on query chunk: %7d, %4d\n", qID, nqrows);

		/* find the neighbors of the chunk */
		for (dID = 0; dID < vault->ndocs; dID += params->ndrows) {
			ndrows = gk_min(params->ndrows, vault->ndocs - dID);
			blkid = dID / params->ndrows;
			hpmat = pmatidx[blkid];
			/* spawn the work threads */
			gk_startwctimer(params->timer_3);
			docnr = hpmat->nrows;
			ddcolptr = dcolptr[blkid];
			ddrowind = drowind[blkid];
			ddrowval = drowval[blkid];
			docnr = hpmat->nrows;
			docnnz = hpmat->rowind[docnr];
			cudaSimilarity<<<nqrows,ndrows,LOAD_QUERY*sizeof(float)/2-2>>>(qID,vault->mat,ddcolptr,ddrowind,ddrowval,
					params->nnbrs,docnr,docnnz,LOAD_QUERY,mynallhits,myallhits);

			gk_stopwctimer(params->timer_3);

		}

		/* write the results in the file */
//		if (fpout) {
//			for (i = 0; i < nqrows; i++) {
//				for (j = 0; j < nallhits[i]; j++) {
//					fprintf(fpout, "%8d %8d %.3f\n", qID + i, allhits[i][j].pid,
//							allhits[i][j].sim.f);
//				}
//			}
//		}
	}

	myallhits = (sim_t **) malloc(sizeof(sim_t*) * params->nqrows);
	mynallhits = (int*) calloc(sizeof(int), params->nqrows);
	for (i = 0; i < params->nqrows; ++i) {
		cudaFree((void**) &myallhits[i]);
	}
	free((void**) myallhits);
	free((void*) mynallhits);

	gk_stopwctimer(params->timer_1);

	/* cleanup and exit */
	if (fpout)
		gk_fclose(fpout);
	gk_FreeMatrix((void ***) &allhits, params->nqrows, 2 * params->nnbrs);
	gk_free((void **) &nallhits, LTERM);

	FreeVault(vault);

	return;
}

__global__ void cudaSimilarity(int qstartid, gk_csr_t *mat, int *dcolptr,
		int *drowind, float *drowval, int docnr, int docnnz, int nnbr,
		int numshared, int *mynallhits, sim_t **myallhits) {
//	int qid = qstartid + blockIdx.x;
//	__shared__
//	float tmpcolval;
//	__shared__
//	int tmpcolind;
//
//	int qnum = mat->rowptr[qid + 1] - mat->rowptr[qid];
//	sim_t *out = myallhits[blockDim.x];
//	int i, j, k;
//	for (k = 0; k < qnum; k += numshared) {
//		for (i = 0; i < qnum; i += blockDim.x) {
//			tmpcolind = mat->colind[k + i];
//			tmpcolval = mat->colval[k + i];
//		}
//		__syncthreads();
//	}

//	int *a=(int*)malloc(34);

}
void cudaFindNeighbors(params_t *params, vault_t *vault, int qID, int nqrows,
		int dID, int ndrows, int *nallhits, sim_t **allhits) {

}

/*************************************************************************/
/*! Computes the neighbors of a set of rows against the documents in
 vault->pmat using OpenMP */
/**************************************************************************/

__device__ void testAdd(int *a, int *b, int *c) {
	*c = *a + *b;
}

int gk_csr_cuda_GetSimilarRows(gk_csr_t *mat, int nqterms, int *qind,
		float *qval, int simtype, int nsim, float minsim, gk_fkv_t *hits,
		int *i_marker, gk_fkv_t *i_cand) {

	return 0;
}

