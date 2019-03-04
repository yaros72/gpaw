/*  This file (wfd.c) is a modified copy of fd.c
 *  with added support for nonlocal operator weights.
 *  The original copyright note of fd.c follows:
 *  Copyright (C) 2003-2007  CAMP
 *  Please see the accompanying LICENSE file for further information. */

#include "bmgs.h"
#include <pthread.h>
#include "../extensions.h"

struct Z(wfds){
  int thread_id;
  int nthds;
  int nweights;
  const bmgsstencil* s;
  const double** w;
  const T* a;
  T* b;
};

void *Z(bmgs_wfd_worker)(void *threadarg)
{
  struct Z(wfds) *args = (struct Z(wfds) *) threadarg;
  const T* a = args->a;
  T* b = args->b;
  const bmgsstencil* stencils = args->s;
  const int n0 = stencils[0].n[0];
  const int n1 = stencils[0].n[1];
  const int n2 = stencils[0].n[2];
  const int j1 = stencils[0].j[1];
  const int j2 = stencils[0].j[2];
  const double** weights = (const double**) GPAW_MALLOC(double*, args->nweights);

  int chunksize = n0 / args->nthds + 1;
  int nstart = args->thread_id * chunksize;
  if (nstart >= n0)
    return NULL;
  int nend = nstart + chunksize;
  if (nend > n0)
    nend = n0;

  for (int i0 = nstart; i0 < nend; i0++)
    {
      const T* aa = a + i0 * (j1 + n1 * (j2 + n2));
      T* bb = b + i0 * n1 * n2;
      for (int iw = 0; iw < args->nweights; iw++)
        weights[iw] = args->w[iw] + i0 * n1 * n2;

      for (int i1 = 0; i1 < n1; i1++)
        {
          for (int i2 = 0; i2 < n2; i2++)
            {
              T x = 0.0;
              for (int iw = 0; iw < args->nweights; iw++)
                {
                  const bmgsstencil* s = &(stencils[iw]);
                  T tmp = 0.0;
                  for (int c = 0; c < s->ncoefs; c++)
                    tmp += aa[s->offsets[c]] * s->coefs[c];
                  tmp *= weights[iw][0];
                  x += tmp;
                  weights[iw]++;
                }
              *bb++ = x;
              aa++;
            }
          aa += j2;
        }
    }
  free(weights);
  return NULL;
}



void Z(bmgs_wfd)(int nweights, const bmgsstencil* stencils, const double** weights, const T* a, T* b)
{
  a += (stencils[0].j[0] + stencils[0].j[1] + stencils[0].j[2]) / 2;

  int nthds = 1;
#ifdef GPAW_OMP_MONLY
  if (getenv("OMP_NUM_THREADS") != NULL)
    nthds = atoi(getenv("OMP_NUM_THREADS"));
#endif
  struct Z(wfds) *wargs = GPAW_MALLOC(struct Z(wfds), nthds);
  pthread_t *thds = GPAW_MALLOC(pthread_t, nthds);

  for(int i=0; i < nthds; i++)
    {
      (wargs+i)->thread_id = i;
      (wargs+i)->nthds = nthds;
      (wargs+i)->nweights = nweights;
      (wargs+i)->s = stencils;
      (wargs+i)->w = weights;
      (wargs+i)->a = a;
      (wargs+i)->b = b;
    }
#ifdef GPAW_OMP_MONLY
  for(int i=1; i < nthds; i++)
    pthread_create(thds + i, NULL, Z(bmgs_wfd_worker), (void*) (wargs+i));
#endif
  Z(bmgs_wfd_worker)(wargs);
#ifdef GPAW_OMP_MONLY
  for(int i=1; i < nthds; i++)
    pthread_join(*(thds+i), NULL);
#endif
  free(wargs);
  free(thds);

}
