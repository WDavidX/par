/*!
\file  main.c
\brief This file is the entry point for paragon's various components
 
\date   Started 11/27/09
\author George
\version\verbatim $Id: omp_main.c 9585 2011-03-18 16:51:51Z karypis $ \endverbatim
*/


#include "simdocs.h"
#include "omp.h"

/*************************************************************************/
/*! Setup the number of threads */
/**************************************************************************/
#ifndef NTHREADS
#define NTHREADS 8
#endif

#ifndef SETROWNUM
//#define SETROWNUM 80
#define SETROWNUM 2147483647
#endif

//#ifndef BLOCKSIZE
//#define BLOCKSIZE 80
//#endif

int blk_rstart[NTHREADS+1];

/*************************************************************************/
/*! This is the entry point for finding simlar patents */
/**************************************************************************/
int main(int argc, char *argv[])
{
  params_t params;
  int rc = EXIT_SUCCESS;
  // set the number of threads to use
  omp_set_num_threads(NTHREADS);
  cmdline_parse(&params, argc, argv);

  printf("********************************************************************************\n");
  printf("sd (%d.%d.%d) Copyright 2011, GK.\n", VER_MAJOR, VER_MINOR, VER_SUBMINOR);
  printf("  nnbrs=%d, minsim=%.2f\n",
      params.nnbrs, params.minsim);

  gk_clearwctimer(params.timer_global);
  gk_clearwctimer(params.timer_1);
  gk_clearwctimer(params.timer_2);
  gk_clearwctimer(params.timer_3);
  gk_clearwctimer(params.timer_4);

  gk_startwctimer(params.timer_global);

  ComputeNeighbors(&params);

  gk_stopwctimer(params.timer_global);

  printf("    wclock: %.2lfs\n", gk_getwctimer(params.timer_global));
  printf("    timer1: %.2lfs\n", gk_getwctimer(params.timer_1));
  printf("    timer2: %.2lfs\n", gk_getwctimer(params.timer_2));
//  printf("    timer3: %.2lfs\n", gk_getwctimer(params.timer_3));
//  printf("    timer4: %.2lfs\n", gk_getwctimer(params.timer_4));
  printf("********************************************************************************\n");

  exit(rc);
}


/*************************************************************************/
/*! Reads and computes the neighbors of each document */
/**************************************************************************/
void ComputeNeighbors(params_t *params)
{
  int i, j, nhits;
  gk_csr_t *mat;
  int32_t *marker;
  gk_fkv_t *hits, *cand;
  FILE *fpout;

  printf("Reading data for %s...\n", params->infstem);

  gk_startwctimer(params->timer_2);
  mat = gk_csr_Read(params->infstem, GK_CSR_FMT_CSR, 1, 0);

  printf("#docs: %d, #nnz: %d.\n", mat->nrows, mat->rowptr[mat->nrows]);

  if (mat->nrows > SETROWNUM){
  	mat->nrows = SETROWNUM;
  	printf("Actual tested nrow %d\n",mat->nrows);
  }

  printf("   OMP thread number %d with total rownum=%d\n",NTHREADS,mat->nrows);


  /* compact the column-space of the matrices */
  gk_csr_CompactColumns(mat);

  /* perform auxiliary normalizations/pre-computations based on similarity */
  gk_csr_Normalize(mat, GK_CSR_ROW, 2);

  /* create the inverted index */
  gk_csr_CreateIndex(mat, GK_CSR_COL);
  gk_stopwctimer(params->timer_2);

  /* create the output file */
  fpout = (params->outfile ? gk_fopen(params->outfile, "w", "ComputeNeighbors: fpout") : NULL);

	int blkavg = mat->rowptr[mat->nrows] / NTHREADS;
	for (i = 0, j = 0; i < mat->nrows; ++i) {
		if (i == mat->nrows - 1) {
			blk_rstart[j + 1] = mat->nrows;
			++j;
			break;
		}
		if (mat->rowptr[i + 1] - mat->rowptr[blk_rstart[j]] > blkavg) {
			blk_rstart[j + 1] = i + 1;
			++j;
		}
	}

//	for (i=0;i<NTHREADS;++i){
//		printf("BLK %d; rstart %d; rend %d; nrow %d\n", i, blk_rstart[i],
//				blk_rstart[i + 1] - 1, blk_rstart[i + 1] - blk_rstart[i]);
//	}

  omp_lock_t filelock;
  omp_init_lock(&filelock);
  gk_startwctimer(params->timer_1);
#pragma omp parallel private(marker, hits, cand,i,nhits)
	{
		int tid = omp_get_thread_num();
//  	printf("In thread %d\n",tid);

		/* allocate memory for the necessary working arrays */
		hits = gk_fkvmalloc(mat->nrows, "ComputeNeighbors: hits");
		marker = gk_i32smalloc(mat->nrows, -1, "ComputeNeighbors: marker");
		cand = gk_fkvmalloc(mat->nrows, "ComputeNeighbors: cand");

		/* find the best neighbors for each query document */

		for (i = blk_rstart[tid]; i < blk_rstart[tid + 1]; i++) {
			if (params->verbosity > 0)
				printf("Working on query %7d\n", i);

	    /* find the neighbors of the ith document */
	    nhits = gk_csr_GetSimilarRows(mat,
	                 mat->rowptr[i+1]-mat->rowptr[i],
	                 mat->rowind+mat->rowptr[i],
	                 mat->rowval+mat->rowptr[i],
	                 GK_CSR_COS, params->nnbrs, params->minsim, hits,
	                 marker, cand);

			/* write the results in the file */


			if (fpout) {
				omp_set_lock(&filelock);
				for (j = 0; j < nhits; j++)
					fprintf(fpout, "%8d %8d %.3f\n", i, hits[j].val, hits[j].key);
				omp_unset_lock(&filelock);
			}

		}
		gk_free((void **) &hits, &marker, &cand, LTERM);
	}
	omp_destroy_lock(&filelock);
	gk_stopwctimer(params->timer_1);

	/* cleanup and exit */
	if (fpout) 		gk_fclose(fpout);

	gk_csr_Free(&mat);

	return;
}

