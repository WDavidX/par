/*!
\file  main.c
\brief This file is the entry point for paragon's various components

\date   Started 11/27/09
\author George
\version\verbatim $Id: omp_main.c 9585 2011-03-18 16:51:51Z karypis $ \endverbatim
*/


#include "simdocs.h"
#include "pthread.h"

/*************************************************************************/
/*! Setup the number of threads */
/**************************************************************************/
#ifndef NTHREADS
#define NTHREADS 8
#endif

typedef	struct{
	gk_csr_t* mat_ptr;
	FILE* fout_ptr;
	params_t* param_ptr;
	int offset;
} threadpack_t;

pthread_t threads[NTHREADS];
int blk_rstart[NTHREADS];
int blk_nrows[NTHREADS];


void *workingthread(void *threadpackptr);
/*************************************************************************/
/*! This is the entry point for finding simlar patents */
/**************************************************************************/
int main(int argc, char *argv[])
{
  params_t params;
  int rc = EXIT_SUCCESS;

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
  printf("    timer3: %.2lfs\n", gk_getwctimer(params.timer_3));
  printf("    timer4: %.2lfs\n", gk_getwctimer(params.timer_4));
  printf("********************************************************************************\n");

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
  FILE *fpout;

  printf("Reading data for %s...\n", params->infstem);

  mat = gk_csr_Read(params->infstem, GK_CSR_FMT_CSR, 1, 0);

  printf("#docs: %d, #nnz: %d.\n", mat->nrows, mat->rowptr[mat->nrows]);

  /* compact the column-space of the matrices */
  gk_csr_CompactColumns(mat);

  /* perform auxiliary normalizations/pre-computations based on similarity */
  gk_csr_Normalize(mat, GK_CSR_ROW, 2);

  /* create the inverted index */
  gk_csr_CreateIndex(mat, GK_CSR_COL);

  /* create the output file */
  fpout = (params->outfile ? gk_fopen(params->outfile, "w", "ComputeNeighbors: fpout") : NULL);

  int kk;
	int blknum=0;
	int partialsum=0, lastpartialsum=0;
	int diff_last,diff_this;
	int blk_avg=mat->rowptr[mat->nrows]/NTHREADS;
	int this_qterm;
	blk_rstart[0]=0;
  for (kk=0;kk<mat->nrows;++kk){
			this_qterm= mat->rowptr[kk+1]-mat->rowptr[kk];
			partialsum+= this_qterm;
			if (partialsum< blk_avg){
					lastpartialsum=partialsum;
			}else{
					diff_last=blk_avg-lastpartialsum;
					diff_this=partialsum-blk_avg;
					/* printf("partialsum %4d; lastpartial %4d; this qterm %4d\t",partialsum,lastpartialsum,this_qterm); */
					if (diff_this>diff_last){
							blk_nrows[blknum]=kk-blk_rstart[blknum];
							++blknum;
							blk_rstart[blknum]=kk+1;
							lastpartialsum=0;
							partialsum=0;
					}else{
							blk_nrows[blknum]=kk-1-blk_rstart[blknum];
							++blknum;
							blk_rstart[blknum]=kk;
							partialsum=this_qterm;
							lastpartialsum=this_qterm;
					}
					/* printf("@%d blk %d(%d/%d), rstart %d, nrows %d\n",kk,blknum-1,blk_avg,mat->nrows, blk_rstart[blknum-1],blk_nrows[blknum-1]); */
			}

	}


  printf("Number of threads %d\n",NTHREADS);
  fflush(stdout);
  /* create pthreads*/
  threadpack_t threadpacks[NTHREADS];
  int threadct;
  /* find the best neighbors for each query document */
  gk_startwctimer(params->timer_1);

  for (threadct=0;threadct<NTHREADS;threadct=threadct+1){
	  threadpacks[threadct].fout_ptr=fpout;
	  threadpacks[threadct].mat_ptr=mat;
	  threadpacks[threadct].param_ptr=params;
	  threadpacks[threadct].offset=threadct;
	  pthread_create(&threads[threadct],NULL,workingthread,(void*)(&threadpacks[threadct]));
  }

  gk_stopwctimer(params->timer_1);

  /* Waint and join all threads */
  void *status;
  int finaljointthread;
  for (threadct=0;threadct<NTHREADS;threadct=threadct+1){
	  finaljointthread = pthread_join(threads[threadct], &status);
	  if (finaljointthread)  perror("Error: return error from pthread_join");
  }
  /* cleanup and exit */
  if (fpout) gk_fclose(fpout);
  gk_csr_Free(&mat);
  return;
}

void *workingthread(void* threadpackptr){
	  int32_t *marker;
	  gk_fkv_t *hits, *cand;
	  threadpack_t* packptr=(threadpack_t*)threadpackptr;

	  gk_csr_t* mat=packptr->mat_ptr;
	  FILE* fpout= packptr->fout_ptr;
	  params_t* params=packptr->param_ptr;

	  int i, nhits, j;
	  int ioffset=packptr->offset;

	  /* allocate memory for the necessary working arrays */
	  hits   = gk_fkvmalloc(mat->nrows, "ComputeNeighbors: hits");
	  marker = gk_i32smalloc(mat->nrows, -1, "ComputeNeighbors: marker");
	  cand   = gk_fkvmalloc(mat->nrows, "ComputeNeighbors: cand");

		int blk_start=blk_rstart[ioffset];
		int blk_end=blk_rstart[ioffset]+blk_nrows[ioffset];
	  for (i=blk_start; i<blk_end; ++i) {
	    if (params->verbosity > 0 )
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
	      for (j=0; j<nhits; j++)
	        fprintf(fpout, "%8d %8d %.3f\n", i, hits[j].val, hits[j].key);
	    }
	  }

	  gk_free((void **)&hits, &marker, &cand, LTERM);
	  pthread_exit(NULL);
}
