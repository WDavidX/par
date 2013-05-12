/*! 
\file 
\brief Data-structure definitions

This file contains various data structures used by paragon 

\date Started 11/27/09
\author George
\version $Id: struct.h 9585 2011-03-18 16:51:51Z karypis $
*/

#ifndef _SIMPAT_STRUCT_H_
#define _SIMPAT_STRUCT_H_


/*************************************************************************/
/*! This data structure stores the similarity scores */
/*************************************************************************/
typedef struct {
  int pid;      /*!< The patent ID */
  union { 
    int i;
    float f;
  } sim;       /*!< The total similarity */
} sim_t;


/*************************************************************************/
/*! This data structure stores a ind/val pair */
/*************************************************************************/
typedef struct {
  int ind;      
  float val;    
} iv_t;


/*************************************************************************/
/*! This data structure stores the patent information for nbrs calculations */
/*************************************************************************/
typedef struct {
  int32_t ndocs;            /*!< The number of documents */
  gk_csr_t *mat;            /*!< The csr-matrix storing the term vectors for abstracts */ 
  gk_csr_t *pmat;           /*!< The csr-matrix storing the term vectors of the current chunk */ 
} vault_t;


/*************************************************************************/
/*! This data structure stores the various variables that make up the 
 * overall state of the system. */
/*************************************************************************/
typedef struct {
  int nnbrs;                    /*!< The maximum number of nearest grants to output */
  int simtype;                  /*!< The type of similarity to use */
  float minsim;                 /*!< The minimum similarity to use for keeping neighbors */ 
  int startid, endid;           /*!< The start/end IDs of the queries */
  int nblocks;                  /*!< The number of cuda blocks to use */
  int nthreads;                 /*!< The number of threads to use */
  int nqrows, ndrows;           /*!< Computation blocking information */
  int usecuda;                  /*!< Indicates cuda-based computations */

  int verbosity;                /*!< The reporting verbosity level */

  char *infstem;                /*!< The filestem of the input file */
  char *outfile;                /*!< The filename where the output will be stored */

  /* timers */
  double timer_global;
  double timer_1;
  double timer_2;
  double timer_3;
  double timer_4;
} params_t;


#endif 
