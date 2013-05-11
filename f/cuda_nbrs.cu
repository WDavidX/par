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

#define CUDABLKNTH 2048
#define CUDABLKSIZE (2048)

#define PASSHERE() do{ fprintf(stderr,"Passline %d\n",__LINE__);}while(0)
#define PINT(x) do{ fprintf(stderr,"Line %d, %s=%d\n",__LINE__,#x,x);}while(0)
#define DPINT(x) do{ int localvaluetoshow;\
		cudaMemcpy((void*) &localvaluetoshow,(void*)&(x),sizeof(int),cudaMemcpyDeviceToHost);\
		fprintf(stderr,"Line %d, %s=%d\n",__LINE__,#x,localvaluetoshow);}while(0)

#define CPHTD(dst,src,num) do{ cudaMemcpy((void*)(dst),(void*) (src), (num),cudaMemcpyHostToDevice);} while(0)
#define CPDTH(dst,src,num) do{ cudaMemcpy((void*)(dst),(void*) (src), (num),cudaMemcpyDeviceToHost);} while(0)

void cudaFindNeighbors(params_t *params, vault_t *vault, int qID, int nqrows,
		int dID, int ndrows, int *nallhits, sim_t **allhits);
int gk_csr_cuda_GetSimilarRows(gk_csr_t *mat, int nqterms, int *qind,
		float *qval, int simtype, int nsim, float minsim, gk_fkv_t *hits,
		int *i_marker, gk_fkv_t *i_cand);

__global__ void cuda_GetSimilarity();
__global__ void copyPmat(gk_csr_t *dmat, gk_csr_t *hmat);
/*************************************************************************/
/*! Top-level routine for computing the neighbors of each document */
/**************************************************************************/
void ompComputeNeighbors(params_t *params) {
	int i, j, qID, dID, nqrows, ndrows;
	vault_t *vault;
//  gk_csr_t *mat;
	sim_t **allhits;
	int *nallhits;
	FILE *fpout;
	params->nqrows=CUDABLKSIZE;
	params->ndrows=CUDABLKNTH-params->nnbrs;
	printf("Reading data for %s...\n", params->infstem);

	vault = ReadData(params);

	params->endid = (params->endid == -1 ? vault->mat->nrows : params->endid);

	printf("#docs: %d, #nnz: %d.\n", vault->ndocs,
			vault->mat->rowptr[vault->mat->nrows]);

	/* Compact the column-space of the matrices */
	gk_csr_CompactColumns(vault->mat);

	/* Perform auxiliary normalizations/pre-computations based on similarity */
	gk_csr_Normalize(vault->mat, GK_CSR_ROW, 2);

	/* create the output file */
	fpout = (
			params->outfile ?
					gk_fopen(params->outfile, "w", "ComputeNeighbors: fpout") :
					NULL);

	/* allocate memory for the working sets */
	gk_AllocMatrix((void ***) &allhits, sizeof(sim_t), params->nqrows,
			2 * params->nnbrs);
	nallhits = gk_imalloc(params->nqrows, "gComputeNeighbors: nallhits");

	gk_startwctimer(params->timer_1);

	omp_set_num_threads(8);

	/* Varibles */
	cudaFree((void*)NULL);
	gk_csr_t *dpmat,*hpmat;
	int *dqcolptr, *dqrowind;
	int qnr,qnnz;
	float *dqrowval;
	
	float **simval;
	int **simidx;
	int outtablesize=CUDABLKNTH;
	simval=(float**)malloc(sizeof(float*)*(params->nqrows));
	simidx=(int**)malloc(sizeof(int*)*(params->nqrows));
	for (i=0;i<params->nqrows;++i){
		cudaMalloc((void**)&simval[i],sizeof(float)* CUDABLKNTH);
		cudaMalloc((void**)&simidx[i],sizeof(int)* CUDABLKNTH);		
	}
	/* break the computations into chunks */
	for (qID = params->startid; qID < params->endid; qID += params->nqrows) {
		nqrows = gk_min(params->nqrows, params->endid - qID);
//		nqrows = gk_min(CUDABLKSIZE, params->endid - qID);
		gk_iset(nqrows, 0, nallhits);

		if (params->verbosity > 0)
			printf("Working on query chunk: %7d, %4d\n", qID, nqrows);

		/* find the neighbors of the chunk */
		for (dID = 0; dID < vault->ndocs; dID += params->ndrows) {
			ndrows = gk_min(params->ndrows, vault->ndocs - dID);
//			ndrows = gk_min(CUDABLKNTH, vault->ndocs - dID);
			/* create the sub-matrices */
			gk_startwctimer(params->timer_2);
			vault->pmat = gk_csr_ExtractSubmatrix(vault->mat, dID, ndrows);
			ASSERT(vault->pmat != NULL);
			gk_stopwctimer(params->timer_2);
			gk_startwctimer(params->timer_4);
			gk_csr_CreateIndex(vault->pmat, GK_CSR_COL);
			gk_stopwctimer(params->timer_4);
			hpmat=vault->pmat;
			if (params->verbosity > 1)
				printf("  Working on db chunk: %7d, %4d, %4.2fMB\n", dID,
						ndrows,
						8.0 * vault->pmat->rowptr[vault->pmat->nrows]
								/ (1024 * 1024));

			
			gk_startwctimer(params->timer_3);
			qnr=hpmat->nrows;
			qnnz=hpmat->rowind[qnr];
			cudaMalloc((void**) &dqcolptr, sizeof(int)*(hpmat->ncols+1));
			cudaMemcpy( dqcolptr, hpmat->colptr,sizeof(int)*(hpmat->ncols+1),CPHD);
			cudaMalloc((void**) &dqrowind, sizeof(int)*(qnnz));
			cudaMemcpy( dqrowind, hpmat->rowind,sizeof(int)*(qnnz),CPHD);
			cudaMalloc((void**) &dqrowval, sizeof(float)*(qnnz));
			cudaMemcpy((void*) dqrowval, (void*)hpmat->rowval,sizeof(float)*(qnnz),CPHD);
			
//			cudaSimilarity<<<nqrows,ndrows>>>(qID,vault->mat,
//											dqcolptr,dqrowind,dqrowval,qnr,qnnz,
//											nallhits,allhits);
			
			fprintf(stderr,"----------------------------\n");
			cudaFree((void*)dqcolptr);
			cudaFree((void*)dqrowind);
			cudaFree((void*)dqrowval);		
			
			gk_stopwctimer(params->timer_3);
			
			cudaFree((void*)dpmat);
			gk_csr_Free(&vault->pmat);
		}

		fprintf(stderr,"---------------------------->>>>>\n");
		/* write the results in the file */
		if (fpout) {
			for (i = 0; i < nqrows; i++) {
				for (j = 0; j < nallhits[i]; j++) {
					fprintf(fpout, "%8d %8d %.3f\n", qID + i, allhits[i][j].pid,
							allhits[i][j].sim.f);
				}
			}
		}
	}
	for (i=0;i<CUDABLKSIZE;++i){
		cudaFree((void**)&simval[i]);
		cudaFree((void**)&simidx[i]);
	}
	free((void*) simval);
	free((void*) simidx);
	gk_stopwctimer(params->timer_1);

	/* cleanup and exit */
	if (fpout)
		gk_fclose(fpout);
	gk_FreeMatrix((void ***) &allhits, params->nqrows, 2 * params->nnbrs);
	gk_free((void **) &nallhits, LTERM);

	FreeVault(vault);

	return;
}

__global__ void copyPmat(gk_csr_t *dmat, gk_csr_t *hmat){
//	printf("\ndmat->nrows=%d\n",dmat->nrows);
	printf("hmat->nrows=%d\n",hmat->nrows);
	printf("hmat->ncols=%d\n",hmat->ncols);
	dmat->nrows=hmat->nrows;
	dmat->ncols=hmat->ncols;


}

__global__ void cuda_GetSimilarity() {

}
;

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

/*

 int gk_csr_GetSimilarRows(gk_csr_t *mat, int nqterms, int *qind, float *qval, 
 int simtype, int nsim, float minsim, gk_fkv_t *hits, int *i_marker,
 gk_fkv_t *i_cand)
 {
 int i, ii, j, k, nrows, ncand;
 int *colptr, *colind, *marker;
 float *colval, *rnorms, mynorm, *rsums, mysum;
 gk_fkv_t *cand;

 if (nqterms == 0)
 return 0;

 nrows  = mat->nrows;
 colptr = mat->colptr;
 colind = mat->colind;
 colval = mat->colval;

 marker = (i_marker ? i_marker : gk_ismalloc(nrows, -1, "gk_csr_SimilarRows: marker"));
 cand   = (i_cand   ? i_cand   : gk_fkvmalloc(nrows, "gk_csr_SimilarRows: cand"));

 switch (simtype) {
 case GK_CSR_COS:
 for (ncand=0, ii=0; ii<nqterms; ii++) {
 i = qind[ii];
 for (j=colptr[i]; j<colptr[i+1]; j++) {
 k = colind[j];
 if (marker[k] == -1) {
 cand[ncand].val = k;
 cand[ncand].key = 0;
 marker[k]       = ncand++;
 }
 cand[marker[k]].key += colval[j]*qval[ii];
 }
 }
 break;

 case GK_CSR_JAC:
 for (ncand=0, ii=0; ii<nqterms; ii++) {
 i = qind[ii];
 for (j=colptr[i]; j<colptr[i+1]; j++) {
 k = colind[j];
 if (marker[k] == -1) {
 cand[ncand].val = k;
 cand[ncand].key = 0;
 marker[k]       = ncand++;
 }
 cand[marker[k]].key += colval[j]*qval[ii];
 }
 }

 rnorms = mat->rnorms;
 mynorm = gk_fdot(nqterms, qval, 1, qval, 1);

 for (i=0; i<ncand; i++)
 cand[i].key = cand[i].key/(rnorms[cand[i].val]+mynorm-cand[i].key);
 break;

 case GK_CSR_MIN:
 for (ncand=0, ii=0; ii<nqterms; ii++) {
 i = qind[ii];
 for (j=colptr[i]; j<colptr[i+1]; j++) {
 k = colind[j];
 if (marker[k] == -1) {
 cand[ncand].val = k;
 cand[ncand].key = 0;
 marker[k]       = ncand++;
 }
 cand[marker[k]].key += gk_min(colval[j], qval[ii]);
 }
 }

 rsums = mat->rsums;
 mysum = gk_fsum(nqterms, qval, 1);

 for (i=0; i<ncand; i++)
 cand[i].key = cand[i].key/(rsums[cand[i].val]+mysum-cand[i].key);
 break;

 default:
 gk_errexit(SIGERR, "Unknown similarity measure %d\n", simtype);
 return -1;
 }

 // go and prune the hits that are bellow minsim 
 for (j=0, i=0; i<ncand; i++) {
 marker[cand[i].val] = -1;
 if (cand[i].key >= minsim) 
 cand[j++] = cand[i];
 }
 ncand = j;

 if (nsim == -1 || nsim >= ncand) {
 nsim = ncand;
 }
 else {
 nsim = gk_min(nsim, ncand);
 gk_dfkvkselect(ncand, nsim, cand);
 gk_fkvsortd(nsim, cand);
 }

 gk_fkvcopy(nsim, cand, hits);

 if (i_marker == NULL)
 gk_free((void **)&marker, LTERM);
 if (i_cand == NULL)
 gk_free((void **)&cand, LTERM);

 return nsim;}
 */