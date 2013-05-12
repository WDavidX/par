/*!
\file  
\brief This file contains function prototypes

\date Started 1/18/07
\author George
\version\verbatim $Id: proto.h 9628 2011-03-23 21:15:43Z karypis $ \endverbatim
*/

#ifndef _SIMPAT_PROTO_H_
#define _SIMPAT_PROTO_H_

#ifdef __cplusplus
extern "C"
{
#endif

/* main.c */
//int main(int argc, char *argv[]);


/* cmdline.c */
void cmdline_parse(params_t *ctrl, int argc, char *argv[]);


/* ompnbrs.c */
void ompComputeNeighbors(params_t *params);
void ompFindNeighbors(params_t *params, vault_t *vault, int qID, int nqrows, 
         int dID, int ndrows, int *nallhits, sim_t **allhits);

/* cudanbrs.c */
void cudaComputeNeighbors(params_t *params);
void cudaComputeNeighbors2(params_t *params);


/* common.c */
vault_t *ReadData(params_t *params);
void FreeVault(vault_t *vault);
void simsortd(size_t n, sim_t *base);


#ifdef __cplusplus
}
#endif

#endif 
