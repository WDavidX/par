/*
 * util.c
 *
 *  Created on: Mar 15, 2013
 *      Author: d
 */

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



/* Taken from Unix Systems Programming, Robbins & Robbins, p37 */
/*
int makeargv(const char *s, const char *delimiters, char ***argvp) {
   int error;
   int i;
   int numtokens;
   const char *snew;
   char *t;

   if ((s == NULL) || (delimiters == NULL) || (argvp == NULL)) {
      errno = EINVAL;
      return -1;
   }
   *argvp = NULL;
   snew = s + strspn(s, delimiters);
   if ((t = malloc(strlen(snew) + 1)) == NULL)
      return -1;
   strcpy(t,snew);
   numtokens = 0;
   if (strtok(t, delimiters) != NULL)
      for (numtokens = 1; strtok(NULL, delimiters) != NULL; numtokens++) ;

   if ((*argvp = malloc((numtokens + 1)*sizeof(char *))) == NULL) {
      error = errno;
      free(t);
      errno = error;
      return -1;
   }

   if (numtokens == 0)
      free(t);
   else {
      strcpy(t,snew);
      **argvp = strtok(t,delimiters);
      for (i=1; i<numtokens; i++)
         *((*argvp) +i) = strtok(NULL,delimiters);
   }
   *((*argvp) + numtokens) = NULL;
   return numtokens;
}

void freemakeargv(char **argv) {
   if (argv == NULL)
      return;
   if (*argv != NULL)
      free(*argv);
   free(argv);
}
*/
