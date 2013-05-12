/*!
\file  common.c
\brief Contains various routines shared by openMP and CUDA
 
\date   Started 2/3/10
\author George
\version\verbatim $Id: all_common.c 9577 2011-03-16 14:48:49Z karypis $ \endverbatim
*/


#include "simdocs.h"

#define MINCOLLEN       3
#define MINROWLEN       20



/**************************************************************************/
/*! Reads the data required for sim calculations */
/**************************************************************************/
vault_t *ReadData(params_t *params)
{
  vault_t *vault;
  gk_csr_t *mat1, *mat2;

  vault = (vault_t *)gk_malloc(sizeof(vault_t), "ReadData: vault");

  /* read the document vectors */
  printf("Reading documents...\n");
  ASSERT(gk_fexists(params->infstem)); 
  vault->mat = gk_csr_Read(params->infstem, GK_CSR_FMT_CSR, 1, 0);

  vault->ndocs = vault->mat->nrows;

  return vault;
}



/**************************************************************************/
/*! Frees the entire vault */
/**************************************************************************/
void FreeVault(vault_t *vault)
{
  gk_csr_Free(&vault->mat);

  gk_free((void **)&vault, LTERM);
}



/*************************************************************************/
/*! Sorts an array of sim_t in decreasing sim order */
/*************************************************************************/
void simsortd(size_t n, sim_t *base)
{
#define sim_gt(a, b) ((a)->sim.f > (b)->sim.f)
    GK_MKQSORT(sim_t, base, n, sim_gt);
#undef sim_gt
}

