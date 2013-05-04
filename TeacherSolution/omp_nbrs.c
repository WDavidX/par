/*!
\file  ompnbrs.c
\brief Computes the nearest neighbors of each document 
 
\date   Started 2/3/10
\author George
\version\verbatim $Id: omp_nbrs.c 9585 2011-03-18 16:51:51Z karypis $ \endverbatim
*/


#include "simdocs.h"

#define NITERS          20


/*************************************************************************/
/*! Top-level routine for computing the neighbors of each document */
/**************************************************************************/
void ompComputeNeighbors(params_t *params)
{
  int i, j, qID, dID, nqrows, ndrows;
  vault_t *vault;
  gk_csr_t *mat;
  sim_t **allhits;
  int *nallhits;
  FILE *fpout;

  printf("Reading data for %s...\n", params->infstem);

  vault = ReadData(params);

  params->endid = (params->endid == -1 ? vault->mat->nrows : params->endid);

  printf("#docs: %d, #nnz: %d.\n", vault->ndocs, vault->mat->rowptr[vault->mat->nrows]);

  /* Compact the column-space of the matrices */
  gk_csr_CompactColumns(vault->mat);

  /* Perform auxiliary normalizations/pre-computations based on similarity */
  gk_csr_Normalize(vault->mat, GK_CSR_ROW, 2);

  /* create the output file */
  fpout = (params->outfile ? gk_fopen(params->outfile, "w", "ComputeNeighbors: fpout") : NULL);

  /* allocate memory for the working sets */
  gk_AllocMatrix((void ***)&allhits, sizeof(sim_t), params->nqrows, 2*params->nnbrs);
  nallhits = gk_imalloc(params->nqrows, "gComputeNeighbors: nallhits");

  gk_startwctimer(params->timer_1);

  omp_set_num_threads(params->nthreads);

  /* break the computations into chunks */
  for (qID=params->startid; qID<params->endid; qID+=params->nqrows) {
    nqrows = gk_min(params->nqrows, params->endid-qID);
    gk_iset(nqrows, 0, nallhits);

    if (params->verbosity > 0)
      printf("Working on query chunk: %7d, %4d\n", qID, nqrows);

    /* find the neighbors of the chunk */ 
    for (dID=0; dID<vault->ndocs; dID+=params->ndrows) {
      ndrows = gk_min(params->ndrows, vault->ndocs-dID);

      /* create the sub-matrices */
      gk_startwctimer(params->timer_2);
      vault->pmat = gk_csr_ExtractSubmatrix(vault->mat, dID, ndrows);
      ASSERT(vault->pmat != NULL);
      gk_stopwctimer(params->timer_2);
      gk_startwctimer(params->timer_4);
      gk_csr_CreateIndex(vault->pmat, GK_CSR_COL);
      gk_stopwctimer(params->timer_4);

      if (params->verbosity > 1)
        printf("  Working on db chunk: %7d, %4d, %4.2fMB\n", dID, ndrows, 
            8.0*vault->pmat->rowptr[vault->pmat->nrows]/(1024*1024));

      /* spawn the work threads */
      gk_startwctimer(params->timer_3);
      ompFindNeighbors(params, vault, qID, nqrows, dID, ndrows, nallhits, allhits);
      gk_stopwctimer(params->timer_3);

      gk_csr_Free(&vault->pmat);
    }

    /* write the results in the file */
    if (fpout) {
      for (i=0; i<nqrows; i++) {
        for (j=0; j<nallhits[i]; j++) {
          fprintf(fpout, "%8d %8d %.3f\n", qID+i, allhits[i][j].pid, allhits[i][j].sim.f);
        }
      }
    }
  }

  gk_stopwctimer(params->timer_1);

  /* cleanup and exit */
  if (fpout) gk_fclose(fpout);
  gk_FreeMatrix((void ***)&allhits, params->nqrows, 2*params->nnbrs);
  gk_free((void **)&nallhits, LTERM);

  FreeVault(vault);

  return;
}



/*************************************************************************/
/*! Computes the neighbors of a set of rows against the documents in
    vault->pmat using OpenMP */
/**************************************************************************/
void ompFindNeighbors(params_t *params, vault_t *vault, int qID, int nqrows, 
         int dID, int ndrows, int *nallhits, sim_t **allhits)
{

  #pragma omp parallel 
  {
    int i, j, k, l, ci, nchits, nhits, nnbrs, noldhits;
    int32_t *marker;
    gk_fkv_t *cand, *chits;
    sim_t *hits, *newhits, *oldhits;

    marker  = gk_i32smalloc(gk_max(params->ndrows, vault->pmat->nrows), -1, "FindNeighbors: marker");
    cand    = gk_fkvmalloc(gk_max(params->ndrows, vault->pmat->nrows), "FindNeighbors: cand");
    chits   = gk_fkvmalloc(gk_max(params->ndrows, vault->pmat->nrows), "FindNeighbors: chits");

    hits    = (sim_t *)gk_malloc(params->ndrows*sizeof(sim_t), "FindNeighbors: hits");

    #pragma omp for schedule(dynamic,NITERS) 
    for (i=0; i<nqrows; i++) {
      /* compute the similarity */
      nhits = gk_csr_GetSimilarRows(vault->pmat, 
                   vault->mat->rowptr[qID+i+1]-vault->mat->rowptr[qID+i], 
                   vault->mat->rowind+vault->mat->rowptr[qID+i], 
                   vault->mat->rowval+vault->mat->rowptr[qID+i], 
                   GK_CSR_COS, params->nnbrs, params->minsim, chits, marker, cand);

      for (k=0; k<nhits; k++) {
        hits[k].pid   = dID + chits[k].val;
        hits[k].sim.f = chits[k].key;
      }


      /* merge with the current best neighbors */
      simsortd(nhits, hits);
      nnbrs = gk_min(params->nnbrs, nhits);

      newhits  = allhits[i];
      oldhits  = newhits + params->nnbrs;
      noldhits = nallhits[i];
      memcpy(oldhits, newhits, sizeof(sim_t)*noldhits);

      /* the two lists to be merged are (nnbrs, hits) and (noldhits, oldhits) */
      for (l=0, j=0, k=0; j<nnbrs && k<noldhits && l<params->nnbrs; l++) {
        if (hits[j].sim.f >= oldhits[k].sim.f) {
          newhits[l] = hits[j];
          j++;
        }
        else {
          newhits[l] = oldhits[k];
          k++;
        }
      }
      for (; j<nnbrs && l<params->nnbrs; l++, j++) 
        newhits[l] = hits[j];
      for (; k<noldhits && l<params->nnbrs; l++, k++) 
        newhits[l] = oldhits[k];
      nallhits[i] = l;

    }

    gk_free((void **)&chits, &marker, &cand, &hits, LTERM);
  }

}




