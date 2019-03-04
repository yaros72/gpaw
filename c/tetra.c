#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "extensions.h"

int compare_doubles (const void *a, const void *b)
{
  const double *da = (const double *) a;
  const double *db = (const double *) b;

  return (*da > *db) - (*da < *db);
}

PyObject* tetrahedron_weight(PyObject *self, PyObject *args)
{
  PyArrayObject* epsilon_k;
  int K;
  PyArrayObject* allsimplices_sk;
  PyArrayObject* simplices_s;
  PyArrayObject* Win_w;
  PyArrayObject* omega_w;
  PyArrayObject* vol_s;

  double f10, f20, f21, f30, f31, f32;
  double f01, f02, f12, f03, f13, f23;
  double omega;
  
  if (!PyArg_ParseTuple(args, "OOiOOOO",
                        &epsilon_k, &allsimplices_sk, &K,
                        &simplices_s, &Win_w, &omega_w,
                        &vol_s))
    return NULL;
  
  int nsimplex = PyArray_DIMS(simplices_s)[0];
  int nw = PyArray_DIMS(omega_w)[0];    
  double* e_k = (double*)PyArray_DATA(epsilon_k);
  double* o_w = (double*)PyArray_DATA(omega_w);
  double* W_w = (double*)PyArray_DATA(Win_w);
  long* s_s = (long*)PyArray_DATA(simplices_s);
  int* alls_sk = (int*)PyArray_DATA(allsimplices_sk);
  double* v_s = (double*)PyArray_DATA(vol_s);
  
  double* et_k = GPAW_MALLOC(double, 4);
  
  double gw = 0;
  double Iw = 0;
  double delta = 0;  
  int relk = 0;
  double ek = 0;
  
  for (int s = 0; s < nsimplex; s++) {
    relk = 0;
    for (int k = 0; k < 4; k++) { 
      et_k[k] = e_k[alls_sk[s_s[s] * 4 + k]];
    }
    ek = e_k[K];
    for (int k = 0; k < 4; k++) {
      if (et_k[k] < ek) {
        relk += 1;
      }
    }
    qsort(et_k, 4, sizeof (double), compare_doubles);
    delta = et_k[3] - et_k[0];
    for (int w = 0; w < nw; w++) {
      Iw = 0;
      gw = 0;
      omega = o_w[w];
      
      f10 = (omega - et_k[0]) / (et_k[1] - et_k[0]);
      f20 = (omega - et_k[0]) / (et_k[2] - et_k[0]);
      f21 = (omega - et_k[1]) / (et_k[2] - et_k[1]);
      f30 = (omega - et_k[0]) / (et_k[3] - et_k[0]);
      f31 = (omega - et_k[1]) / (et_k[3] - et_k[1]);
      f32 = (omega - et_k[2]) / (et_k[3] - et_k[2]);
      
      f01 = 1 - f10;
      f02 = 1 - f20;
      f03 = 1 - f30;
      f12 = 1 - f21;
      f13 = 1 - f31;
      f23 = 1 - f32;
      
      if (et_k[1] != et_k[0] && et_k[0] <= omega && omega <= et_k[1])
        {
          gw = 3 * f20 * f30 / (et_k[1] - et_k[0]);
          switch (relk) {
          case 0:
            Iw = (f01 + f02 + f03) / 3;
            break;
          case 1:
            Iw = f10 / 3;
            break;
          case 2:
            Iw = f20 / 3;
            break;
          case 3:
            Iw = f30 / 3;
            break;
          }
        }
      else if (et_k[1] != et_k[2] && et_k[1] < omega && omega < et_k[2])
        {
          gw = 3 / delta * (f12 * f20 + f21 * f13);
          switch (relk) {
          case 0:
            Iw = f03 / 3 + f02 * f20 * f12 / (gw * delta);
            break;
          case 1:
            Iw = f12 / 3 + f13 * f13 * f21 / (gw * delta);
            break;
          case 2:
            Iw = f21 / 3 + f20 * f20 * f12 / (gw * delta);
            break;
          case 3:
            Iw = f30 / 3 + f31 * f13 * f21 / (gw * delta);
            break;
          }
        }
      else if (et_k[2] != et_k[3] && et_k[2] <= omega && omega <= et_k[3])
        {
          gw = 3 * f03 * f13 / (et_k[3] - et_k[2]);
          switch (relk) {
          case 0:
            Iw = f03 / 3;
            break;
          case 1:
            Iw = f13 / 3;
            break;
          case 2:
            Iw = f23 / 3;
            break;
          case 3:
            Iw = (f30 + f31 + f32) / 3;
            break;
          }
        }
      else {
        continue;
      }
      W_w[w] += v_s[s] * Iw * gw;
    }
  }
  free(et_k);
  Py_RETURN_NONE;
}
