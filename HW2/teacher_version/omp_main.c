/*!
\file  main.c
\brief This file is the entry point for paragon's various components
 
\date   Started 11/27/09
\author George
\version\verbatim $Id: omp_main.c 9585 2011-03-18 16:51:51Z karypis $ \endverbatim
*/


#include "simdocs.h"


/*************************************************************************/
/*! This is the entry point for finding simlar patents */
/**************************************************************************/
int main(int argc, char *argv[])
{
  params_t params;
  int rc = EXIT_SUCCESS;

  cmdline_parse(&params, argc, argv);

  printf("********************************************************************************\n");
  printf("ompsd (%d.%d.%d) Copyright 2011, GK.\n", VER_MAJOR, VER_MINOR, VER_SUBMINOR);
  printf("  nnbrs=%d, minsim=%.2f, nblocks=%d, nthreads=%d, nqrows=%d, ndrows=%d\n",
      params.nnbrs, params.minsim, params.nblocks, params.nthreads, params.nqrows, params.ndrows);
  printf("  startid=%d, endid=%d\n",
      params.startid, params.endid);

  gk_clearwctimer(params.timer_global);
  gk_clearwctimer(params.timer_1);
  gk_clearwctimer(params.timer_2);
  gk_clearwctimer(params.timer_3);
  gk_clearwctimer(params.timer_4);

  gk_startwctimer(params.timer_global);

//  if (params.usecuda)
//    cudaComputeNeighbors(&params);
//  else
    ompComputeNeighbors(&params);

  gk_stopwctimer(params.timer_global);

  printf("    wclock: %.2lfs\n", gk_getwctimer(params.timer_global));
  printf("    timer1: %.2lfs\n", gk_getwctimer(params.timer_1));
  printf("    timer2: %.2lfs\n", gk_getwctimer(params.timer_2));
  printf("    timer3: %.2lfs\n", gk_getwctimer(params.timer_3));
  printf("    timer4: %.2lfs\n", gk_getwctimer(params.timer_4));
  printf("********************************************************************************\n");

  exit(rc);
}

