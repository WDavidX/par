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

//#ifndef TESTROWNUM
//#define TESTROWNUM 8580
////#define TESTROWNUM 40
//#endif

typedef	struct{
	gk_csr_t* mat_ptr;
	params_t* params_ptr;

	int blk,refrow;

	gk_fkv_t *hits, *cand;
	int32_t *marker;

	int blk_nhits;

} threadpack_t;

pthread_t threads[NTHREADS];
threadpack_t threadpack[NTHREADS];
gk_csr_t* blk_matptr[NTHREADS];
int blk_rstart[NTHREADS+1],blk_nrows[NTHREADS];

int blocksize,lastblocksize;
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
  printf("  Number of threads %d\n",
		  NTHREADS);
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

//  /* allocate memory for the necessary working arrays */
//  hits   = gk_fkvmalloc(mat->nrows, "ComputeNeighbors: hits");
//  marker = gk_i32smalloc(mat->nrows, -1, "ComputeNeighbors: marker");
//  cand   = gk_fkvmalloc(mat->nrows, "ComputeNeighbors: cand");

  /* testing for small for block operation*/
//  mat->nrows=TESTROWNUM;

  gk_startwctimer(params->timer_2);
  int blknnzavg = mat->rowptr[mat->nrows] / NTHREADS;
  blk_rstart[0]=0;
  for (i = 0, j = 0; i < mat->nrows; ++i) {
	if (i == mat->nrows - 1) {
		blk_rstart[j + 1] = mat->nrows;
		blk_nrows[j]=blk_rstart[j + 1]-blk_rstart[j ];
//		fprintf(stderr, "i=%3d j=%3d;   blkrst %3d blknrow %3d , avg-%d |   nnz tot %5d num %5d , avg-%d; \n",i,j,
//			blk_rstart[j]-blk_rstart[0],
//			blk_nrows[j],
//			mat->nrows/NTHREADS,
//			mat->rowptr[blk_rstart[j]] - mat->rowptr[0],
//			mat->rowptr[blk_rstart[j+1]] - mat->rowptr[blk_rstart[j]],
//			blknnzavg);
			blk_matptr[j] = gk_csr_ExtractSubmatrix(mat, blk_rstart[j],
					i + 1 - blk_rstart[j]);
			gk_csr_CreateIndex(blk_matptr[j], GK_CSR_COL);

		++j;
		break;
		}
	if (mat->rowptr[i + 1] - mat->rowptr[blk_rstart[j]] > blknnzavg) {
		blk_rstart[j + 1] = i + 1;
		blk_nrows[j]=blk_rstart[j + 1]-blk_rstart[j ];
		blk_matptr[j] = gk_csr_ExtractSubmatrix(mat, blk_rstart[j],
					i + 1 - blk_rstart[j]);
			gk_csr_CreateIndex(blk_matptr[j], GK_CSR_COL);
//			fprintf(stderr, "i=%3d j=%3d;   blkrst %3d blknrow %3d , avg-%d |   nnz tot %5d num %5d , avg-%d; \n",i,j,
//				blk_rstart[j]-blk_rstart[0],
//				blk_nrows[j],
//				mat->nrows/NTHREADS,
//				mat->rowptr[blk_rstart[j]] - mat->rowptr[0],
//				mat->rowptr[blk_rstart[j+1]] - mat->rowptr[blk_rstart[j]],
//				blknnzavg);

		++j;
		}
	}

  printf("NTHREADS %d\n",j);
  gk_stopwctimer(params->timer_2);
  /* find the best neighbors for each query document */
  gk_startwctimer(params->timer_1);


  /* Variables for block operation*/
  int blknhits[NTHREADS];
  int blk;
  int rownhits_final;
  int rowtothits;
  gk_fkv_t* rowhits=malloc(sizeof(gk_fkv_t)*params->nnbrs*NTHREADS);

  if (!rowhits){
//	  fprintf(stderr,"Failed to malloc rowhits");
	  perror("Failed to malloc rowhits");
	  exit(1);
  }
  int finaljointthread;

  /* Testing submat*/
  /* allocate memory for the necessary working arrays */
  hits   = gk_fkvmalloc(mat->nrows, "ComputeNeighbors: hits");
  marker = gk_i32smalloc(mat->nrows, -1, "ComputeNeighbors: marker");
  cand   = gk_fkvmalloc(mat->nrows, "ComputeNeighbors: cand");

  for (blk = 0; blk < NTHREADS; ++blk) {
		threadpack[blk].blk = blk;
		threadpack[blk].hits = &hits[blk_rstart[blk]];
		threadpack[blk].cand = &cand[blk_rstart[blk]];
		threadpack[blk].marker = &marker[blk_rstart[blk]];
		threadpack[blk].params_ptr = params;
		threadpack[blk].mat_ptr = mat;
	}

  /* Taks assighment */
  for (i=0; i<mat->nrows; ++i) {
    if (params->verbosity > 0)
      printf("\nWorking on query %7d\n", i);



    for (blk=0;blk<NTHREADS;++blk){
    	threadpack[blk].refrow=i;
    	pthread_create(&threads[blk],NULL,workingthread,(void*)(&threadpack[blk]));
    }

    for (blk=0;blk<NTHREADS;++blk){
  	  finaljointthread = pthread_join(threads[blk], NULL);
	  if (finaljointthread)  perror("Error: return error from pthread_join ");
	  blknhits[blk]=threadpack[blk].blk_nhits;
    }

    rowtothits=0;
    for (blk=0;blk<NTHREADS;++blk){
	  for (j=0;j<blknhits[blk];++j){
				rowhits[rowtothits] = (threadpack[blk].hits)[j];
				rowhits[rowtothits].val = rowhits[rowtothits].val
						+ blk_rstart[blk];
				++rowtothits;
	  }
    }

    gk_fkvsortd(rowtothits,rowhits);
    if (rowtothits>params->nnbrs){
    	rownhits_final=params->nnbrs;
    }else{
    	rownhits_final=rowtothits;
    }


    /* write the results in the file */
    if (fpout) {
      for (j=0; j<rownhits_final; j++)
        fprintf(fpout, "%8d %8d %.3f\n", i, rowhits[j].val, rowhits[j].key);
    }

  }
  gk_free((void **)&hits, &marker, &cand, LTERM);
  gk_free((void **)&rowhits, LTERM);

  gk_stopwctimer(params->timer_1);


  /* cleanup and exit */
  if (fpout) gk_fclose(fpout);

  gk_free((void **)&hits, &marker, &cand, LTERM);

  gk_csr_Free(&mat);
  fprintf(stdout,"sd accomplished.\n");
  return;
}

pthread_mutex_t threadcheck = PTHREAD_MUTEX_INITIALIZER;

void *workingthread(void *threadpackptr){
	threadpack_t* packptr=(threadpack_t *) threadpackptr;
	int blk=packptr->blk;
	int i=packptr->refrow;
	params_t* params=packptr->params_ptr;
	gk_csr_t* mat=packptr->mat_ptr;

	packptr->blk_nhits = gk_csr_GetSimilarRows(blk_matptr[blk], //	gk_csr_t* submat=blk_matptr[blk];
		                 mat->rowptr[i+1] - mat->rowptr[i],
		                 mat->rowind + mat->rowptr[i],
		                 mat->rowval + mat->rowptr[i],
		                 GK_CSR_COS, params->nnbrs, params->minsim, packptr->hits,
		                 packptr->marker, packptr->cand);
//	fprintf(stderr,"\t-->BLK %d \t RefRow %d \t nhits %d\n",blk,i,packptr->blk_nhits);
	pthread_exit(NULL);
}

