/*  This file (wrelax.c) is a modified copy of relax.c
 *  with added support for nonlocal operator weights.
 *  The original copyright note of relax.c follows:
 *  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2005       CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#include "bmgs.h"

void bmgs_wrelax(const int relax_method, const int nweights,
                 const bmgsstencil* stencils, const double** weights,
                 double* a, double* b,
                 const double* src, const double w)
{

const int n0 = stencils[0].n[0];
const int n1 = stencils[0].n[1];
const int n2 = stencils[0].n[2];
const int j0 = stencils[0].j[0];
const int j1 = stencils[0].j[1];
const int j2 = stencils[0].j[2];

a += (j0 + j1 + j2) / 2;

if (relax_method == 1)
{
     /* Weighted Gauss-Seidel relaxation for the equation "operator" b = src
        a contains the temporary array holding also the boundary values. */

  for (int i0 = 0; i0 < n0; i0++)
    {
      for (int i1 = 0; i1 < n1; i1++)
        {
          for (int i2 = 0; i2 < n2; i2++)
            {
              double x = 0.0;
              double coef = 0.0;
              for (int iw = 0; iw < nweights; iw++)
                {
                  double weight = weights[iw][0];
                  double tmp = 0.0;
                  const bmgsstencil* s = &(stencils[iw]);
                  for (int c = 1; c < s->ncoefs; c++)
                    tmp += a[s->offsets[c]] * s->coefs[c];
                  tmp *= weight;
                  x += tmp;
                  coef += weight * s->coefs[0];
                  weights[iw]++;
                }
              x = (*src - x) / coef;
              *b++ = x;
              *a++ = x;
              src++;
            }
          a += j2;
        }
      a += j1;
    }

}
else
{
     /* Weighted Jacobi relaxation for the equation "operator" b = src
        a contains the temporariry array holding also the boundary values. */

  double temp;
  for (int i0 = 0; i0 < n0; i0++)
    {
      for (int i1 = 0; i1 < n1; i1++)
        {
          for (int i2 = 0; i2 < n2; i2++)
            {
              double x = 0.0;
              double coef = 0.0;
              for (int iw = 0; iw < nweights; iw++)
                {
                  double weight = weights[iw][0];
                  double tmp = 0.0;
                  const bmgsstencil* s = &(stencils[iw]);
                  for (int c = 1; c < s->ncoefs; c++)
                    tmp += a[s->offsets[c]] * s->coefs[c];
                  tmp *= weight;
                  x += tmp;
                  coef += weight * s->coefs[0];
                  weights[iw]++;
                }
              temp = (1.0 - w) * *b + w * (*src - x) / coef;
              *b++ = temp;
              a++;
              src++;
            }
          a += j2;
        }
      a += j1;
    }
}

}
