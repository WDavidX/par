#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <errno.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <signal.h>
#include <setjmp.h>
#include <assert.h>
#include <inttypes.h>
//#include <sys/resource.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

typedef struct {
  size_t nlines;                    /*!< The maximum number of nearest grants to output */
  char *infile;                /*!< The filestem of the input file */
  char *outfile;                /*!< The filename where the output will be stored */
  double timer_global;
  double timer_1;
  double timer_2;
  double timer_3;
  double timer_4;
  int nthreads,nlevels,nalloc;
  int *a;					/* Store the calculated exclusive */
  int *b;
  int *c;
  int *d;
  char *f,*fb;
} params_t;


double gk_WClockSeconds(void)
{
#ifdef __GNUC__
  struct timeval ctime;

  gettimeofday(&ctime, NULL);

  return (double)ctime.tv_sec + (double).000001*ctime.tv_usec;
#else
  return (double)time(NULL);
#endif
}
#define gk_clearwctimer(tmr) (tmr = 0.0)
#define gk_startwctimer(tmr) (tmr -= gk_WClockSeconds())
#define gk_stopwctimer(tmr)  (tmr += gk_WClockSeconds())
#define gk_getwctimer(tmr)   (tmr)

